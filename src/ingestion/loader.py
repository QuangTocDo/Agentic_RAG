"""
Document loader — load legal .txt files from a directory.
Each file is read in full and returned as a Document-like dict.
"""
import os
from pathlib import Path


def load_text_file(file_path: str) -> dict:
    """Load a single .txt file and return a document dict."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "page_content": content,
        "metadata": {
            "source": str(path.resolve()),
            "filename": path.name,
            "law_name": _extract_law_name(content),
        },
    }


def load_directory(dir_path: str, extensions: tuple[str, ...] = (".txt",)) -> list[dict]:
    """Load all supported files from a directory recursively."""
    docs = []
    dir_p = Path(dir_path)
    if not dir_p.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    for root, _, files in os.walk(dir_p):
        for fname in sorted(files):
            if fname.lower().endswith(extensions):
                fpath = os.path.join(root, fname)
                docs.append(load_text_file(fpath))
    return docs


# --------------- helpers ---------------

def _extract_law_name(text: str) -> str:
    """Try to extract the law name from the first non-empty line."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return "Unknown"


def load_hf_dataset(doc_ids: list[int] | None = None, limit: int = 1000) -> list[dict]:
    """
    Connect to Hugging Face repo 'th1nhng0/vietnamese-legal-documents'
    and load documents by their IDs or up to a limit.
    Use limit <= 0 to load all documents when doc_ids is not provided.
    
    Returns a list of document dicts.
    """
    import os
    import pandas as pd
    import pyarrow.parquet as pq
    import fsspec
    from configs.setting import settings

    token = settings.hf_token
    if not token or token == "your_huggingface_token_here":
        token = os.environ.get("HF_TOKEN", "")

    repo_id = "th1nhng0/vietnamese-legal-documents"
    metadata_url = f"hf://datasets/{repo_id}/data/metadata.parquet"
    content_url = f"hf://datasets/{repo_id}/data/content.parquet"
    
    storage_options = {"token": token} if token else {}
    
    print(f"🔗 Đang kết nối tới Hugging Face Dataset: {repo_id}...")
    try:
        fs = fsspec.filesystem("hf", **storage_options)
        
        # 1. Read metadata.parquet
        print("📂 Đang tải metadata.parquet...")
        with fs.open(metadata_url, "rb") as f:
            df_meta = pd.read_parquet(f)
        
        # Filter metadata for requested IDs, take head(limit), or load all with limit <= 0.
        if doc_ids:
            ids_set = set(doc_ids)
            filtered_meta = df_meta[df_meta["id"].isin(ids_set)]
        else:
            if limit and limit > 0:
                print(f"   ℹ️ Không có ID nào được chỉ định, sẽ tải {limit} tài liệu đầu tiên.")
                filtered_meta = df_meta.head(limit)
            else:
                print("   ℹ️ Không có ID nào được chỉ định, sẽ tải toàn bộ tài liệu.")
                filtered_meta = df_meta
            doc_ids = filtered_meta["id"].tolist()

        print(f"   Đã tìm thấy {len(filtered_meta)} dòng metadata tương ứng.")
        
        if filtered_meta.empty:
            print("   ⚠️ Không tìm thấy metadata hợp lệ.")
            return []
            
        # 2. Read content.parquet
        print("📂 Đang tải content.parquet (đọc có bộ lọc)...")
        with fs.open(content_url, "rb") as f:
            pf = pq.ParquetFile(f)
            
            str_ids = {str(i) for i in doc_ids}
            matching_rows = []
            
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx, columns=["id", "content_html"])
                df_rg = rg.to_pandas()
                df_match = df_rg[df_rg["id"].astype(str).isin(str_ids)]
                if not df_match.empty:
                    matching_rows.append(df_match)
            
            if not matching_rows:
                print("   ⚠️ Không tìm thấy nội dung văn bản nào khớp trong content.parquet.")
                return []
                
            df_content = pd.concat(matching_rows, ignore_index=True)
            df_content = df_content.drop_duplicates(subset=["id"])
            print(f"   Đã tìm thấy {len(df_content)} nội dung văn bản tương ứng (sau khi khử trùng).")
            
        # Merge metadata and content on ID
        df_content["id"] = df_content["id"].astype(int)
        merged = pd.merge(filtered_meta, df_content, on="id", how="inner")
        
        docs = []
        for _, row in merged.iterrows():
            doc_id = int(row["id"])
            title = str(row["title"])
            so_ky_hieu = str(row["so_ky_hieu"])
            tinh_trang_hieu_luc = str(row["tinh_trang_hieu_luc"])
            content_html = str(row["content_html"])
            
            docs.append({
                "page_content": content_html,
                "metadata": {
                    "source": f"hf://{repo_id}/{doc_id}",
                    "filename": f"{doc_id}.html",
                    "law_name": title,
                    "doc_id": doc_id,
                    "so_ky_hieu": so_ky_hieu,
                    "tinh_trang_hieu_luc": tinh_trang_hieu_luc
                }
            })
            
        return docs
        
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu từ Hugging Face: {e}")
        raise e
