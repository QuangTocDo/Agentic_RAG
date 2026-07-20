"""
Document loader — load legal .txt files from a directory or stream from Hugging Face.
Each file is returned as a Document-like dict with standardized metadata.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Generator


def _clean_val(val, default="unknown") -> str:
    """Helper to clean metadata values and handle NaN/nulls."""
    import pandas as pd
    if pd.isna(val) or val is None or str(val).strip() in {"", "nan", "NaN", "None", "..."}:
        return default
    return str(val).strip()


def _extract_local_metadata(content: str, file_path: Path) -> dict:
    """Extract metadata from local legal text files using heuristics and regex."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    
    # 1. law_name (title)
    law_name = lines[0] if lines else "unknown"
    
    # 2. so_ky_hieu (e.g. 91/2015/QH13)
    so_ky_hieu = "unknown"
    if len(lines) > 1:
        match = re.search(r"(?:Luật số|Số|Nghị định số|Quyết định số)\s*[:\s]*([^\)]+)", lines[1], re.IGNORECASE)
        if match:
            so_ky_hieu = match.group(1).strip()
        else:
            for l in lines[:5]:
                match = re.search(r"Số\s*[:\s]*([0-9]+/[0-9]+/QH[0-9]+|[0-9]+/[0-9]+/[A-Z-]+)", l, re.IGNORECASE)
                if match:
                    so_ky_hieu = match.group(1).strip()
                    break
    
    # 3. loai_van_ban
    loai_van_ban = "unknown"
    title_lower = law_name.lower()
    if "bộ luật" in title_lower:
        loai_van_ban = "Bộ luật"
    elif "luật" in title_lower:
        loai_van_ban = "Luật"
    elif "nghị định" in title_lower:
        loai_van_ban = "Nghị định"
    elif "nghị quyết" in title_lower:
        loai_van_ban = "Nghị quyết"
    elif "thông tư" in title_lower:
        loai_van_ban = "Thông tư"
    elif "quuyết định" in title_lower:
        loai_van_ban = "Quyết định"

    # 4. ngay_ban_hanh, ngay_co_hieu_luc, co_quan_ban_hanh
    ngay_ban_hanh = "unknown"
    ngay_co_hieu_luc = "unknown"
    co_quan_ban_hanh = "unknown"
    
    for l in lines[:10]:
        if "ban hành" in l.lower() or "thông qua ngày" in l.lower():
            date_match = re.search(r"ngày\s+(\d+)\s+tháng\s+(\d+)\s+năm\s+(\d+)", l, re.IGNORECASE)
            if date_match:
                ngay_ban_hanh = f"{date_match.group(1).zfill(2)}/{date_match.group(2).zfill(2)}/{date_match.group(3)}"
        if "hiệu lực" in l.lower() and "từ ngày" in l.lower():
            date_match = re.search(r"ngày\s+(\d+)\s+tháng\s+(\d+)\s+năm\s+(\d+)", l, re.IGNORECASE)
            if date_match:
                ngay_co_hieu_luc = f"{date_match.group(1).zfill(2)}/{date_match.group(2).zfill(2)}/{date_match.group(3)}"
                
        if "quốc hội" in l.lower():
            co_quan_ban_hanh = "Quốc hội"
        elif "chính phủ" in l.lower():
            co_quan_ban_hanh = "Chính phủ"
        elif "bộ" in l.lower() and co_quan_ban_hanh == "unknown":
            match = re.search(r"Bộ\s+([A-Za-zĐđÀ-ỹ\s]+)", l)
            if match:
                co_quan_ban_hanh = f"Bộ {match.group(1).strip()}"

    return {
        "source": str(file_path.resolve()),
        "filename": file_path.name,
        "law_name": law_name,
        "so_ky_hieu": so_ky_hieu,
        "loai_van_ban": loai_van_ban,
        "ngay_ban_hanh": ngay_ban_hanh,
        "ngay_co_hieu_luc": ngay_co_hieu_luc,
        "ngay_het_hieu_luc": "unknown",
        "tinh_trang_hieu_luc": "Còn hiệu lực",
        "co_quan_ban_hanh": co_quan_ban_hanh,
        "nguon_thu_thap": "local"
    }


def load_text_file(file_path: str) -> dict:
    """Load a single .txt file and return a document dict with standardized metadata."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "page_content": content,
        "metadata": _extract_local_metadata(content, path),
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


def stream_directory(dir_path: str, extensions: tuple[str, ...] = (".txt",)) -> Generator[dict, None, None]:
    """Yield all supported files from a directory recursively for streaming."""
    dir_p = Path(dir_path)
    if not dir_p.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    for root, _, files in os.walk(dir_p):
        for fname in sorted(files):
            if fname.lower().endswith(extensions):
                fpath = os.path.join(root, fname)
                try:
                    yield load_text_file(fpath)
                except Exception as e:
                    print(f"⚠️ Không thể đọc file {fname}: {e}")


def load_hf_dataset(doc_ids: list[int] | None = None, limit: int = 1000) -> list[dict]:
    """
    Connect to Hugging Face repo 'th1nhng0/vietnamese-legal-documents'
    and load documents by their IDs or up to a limit.
    Use limit <= 0 to load all documents when doc_ids is not provided.
    
    Returns a list of document dicts.
    """
    return list(stream_hf_dataset(doc_ids, limit))


def stream_hf_dataset(doc_ids: list[int] | None = None, limit: int = 1000) -> Generator[dict, None, None]:
    """
    Connect to Hugging Face repo 'th1nhng0/vietnamese-legal-documents'
    and yield documents by their IDs or up to a limit in a memory-efficient generator.
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
            return
            
        # Create metadata lookup by id
        meta_lookup = filtered_meta.set_index("id").to_dict(orient="index")
        str_ids = {str(i) for i in doc_ids}
        yielded_ids = set()
        
        # 2. Read content.parquet (streaming row groups)
        print("📂 Đang tải content.parquet (đọc có bộ lọc theo row group)...")
        with fs.open(content_url, "rb") as f:
            pf = pq.ParquetFile(f)
            
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx, columns=["id", "content_html"])
                df_rg = rg.to_pandas()
                df_match = df_rg[df_rg["id"].astype(str).isin(str_ids)]
                
                if not df_match.empty:
                    for _, row in df_match.iterrows():
                        row_id = int(row["id"])
                        if row_id in yielded_ids:
                            continue
                        yielded_ids.add(row_id)
                        
                        meta = meta_lookup.get(row_id)
                        if not meta:
                            continue
                        
                        title = _clean_val(meta.get("title", ""))
                        so_ky_hieu = _clean_val(meta.get("so_ky_hieu", ""))
                        loai_van_ban = _clean_val(meta.get("loai_van_ban", ""))
                        ngay_ban_hanh = _clean_val(meta.get("ngay_ban_hanh", ""))
                        ngay_co_hieu_luc = _clean_val(meta.get("ngay_co_hieu_luc", ""))
                        ngay_het_hieu_luc = _clean_val(meta.get("ngay_het_hieu_luc", ""))
                        tinh_trang_hieu_luc = _clean_val(meta.get("tinh_trang_hieu_luc", ""))
                        co_quan_ban_hanh = _clean_val(meta.get("co_quan_ban_hanh", ""))
                        nguon_thu_thap = _clean_val(meta.get("nguon_thu_thap", "Hugging Face"))
                        
                        yield {
                            "page_content": str(row["content_html"]),
                            "metadata": {
                                "source": f"hf://{repo_id}/{row_id}",
                                "filename": f"{row_id}.html",
                                "law_name": title,
                                "doc_id": row_id,
                                "so_ky_hieu": so_ky_hieu,
                                "loai_van_ban": loai_van_ban,
                                "ngay_ban_hanh": ngay_ban_hanh,
                                "ngay_co_hieu_luc": ngay_co_hieu_luc,
                                "ngay_het_hieu_luc": ngay_het_hieu_luc,
                                "tinh_trang_hieu_luc": tinh_trang_hieu_luc,
                                "co_quan_ban_hanh": co_quan_ban_hanh,
                                "nguon_thu_thap": nguon_thu_thap
                            }
                        }
                        
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu từ Hugging Face: {e}")
        raise e
