"""
从京剧PDF的「主要角色」版块提取服饰描述，写入对应JSON的 characters[].costume_note
用法：python3 extract_costumes.py
"""
import pdfplumber, re, json, glob, os, sys

PDF_DIR    = os.path.join(os.path.dirname(__file__), '京剧剧本')
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), 'structured', 'scripts')

def parse_costume_section(text: str) -> dict:
    """解析「主要角色」块，返回 {角色名: 服饰描述}"""
    m = re.search(
        r'主要角色\s*\n(.*?)(?=情节|根据|【第|（第|第一场|$)',
        text, re.DOTALL
    )
    if not m:
        return {}
    block = m.group(1).strip()
    result = {}
    current_name, current_val = None, ''
    for line in block.split('\n'):
        line = line.strip()
        if not line:
            continue
        # 新角色行：名字（≤8字）后接中文冒号或英文冒号
        m2 = re.match(r'^([^：:，,]{1,8})[：:](.+)$', line)
        if m2:
            if current_name:
                val = current_val.strip('。，. ')
                if val:
                    result[current_name] = val
            current_name = m2.group(1).strip()
            rest = m2.group(2).strip()
            # 去掉行当前缀：「净，服饰...」或「净：服饰...」
            # 若 m3 不匹配，说明这行只有行当没有服饰描述，current_val 置空
            m3 = re.match(r'^[^，,：:]{1,6}[，,：:](.+)$', rest)
            current_val = m3.group(1).strip() if m3 else ''
        else:
            if current_name:
                current_val += line   # 续行拼接
    if current_name:
        val = current_val.strip('。，. ')
        if val:  # 只存有实际内容的
            result[current_name] = val
    return result

def build_pdf_map():
    """返回 {script_id: pdf_path}"""
    m = {}
    for fp in glob.glob(f'{PDF_DIR}/**/*.pdf', recursive=True):
        hit = re.match(r'^(\d+)', os.path.basename(fp))
        if hit:
            m[hit.group(1)] = fp
    return m

def main():
    pdf_map = build_pdf_map()
    json_files = sorted(glob.glob(f'{SCRIPTS_DIR}/*.json'))
    total = len(json_files)

    files_updated = 0
    chars_updated = 0
    pdfs_with_section = 0

    for i, jpath in enumerate(json_files, 1):
        if i % 100 == 0:
            print(f'  {i}/{total} ...', flush=True)

        with open(jpath, encoding='utf-8') as f:
            try: d = json.load(f)
            except: continue

        sid = d.get('id', '')
        pdf_path = pdf_map.get(sid)
        if not pdf_path:
            continue

        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = pdf.pages[0].extract_text() or ''
        except Exception as e:
            print(f'  [WARN] {sid}: {e}')
            continue

        costumes = parse_costume_section(text)
        if not costumes:
            continue
        pdfs_with_section += 1

        changed = False
        for ch in d.get('characters', []):
            name = ch.get('name', '')
            if name in costumes:
                new_note = costumes[name]
                if ch.get('costume_note') != new_note:
                    ch['costume_note'] = new_note
                    chars_updated += 1
                    changed = True

        if changed:
            files_updated += 1
            with open(jpath, 'w', encoding='utf-8') as f:
                json.dump(d, f, ensure_ascii=False, indent=2)

    print(f'\n=== 完成 ===')
    print(f'有主要角色版块的PDF : {pdfs_with_section}')
    print(f'更新的JSON文件      : {files_updated}')
    print(f'写入costume_note    : {chars_updated} 条')

if __name__ == '__main__':
    main()
