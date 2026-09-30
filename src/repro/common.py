# common.py
# 역할: 재현 스크립트들이 함께 쓰는 작은 도구(파일 해시, vendoring 파일 무결성 검사, git 상태 조회).
# 사용법: 같은 폴더의 스크립트에서 `import common` 후 함수를 호출한다. 단독 실행하면 무결성 검사만 한다.
"""재현 스크립트 공용 도구 모듈.

기존 스크립트(reproduction_*.py 등)는 import하는 순간 실행되는 부작용이 있으므로 여기서 가져다 쓰지 않고,
필요한 로직만 짧게 새로 작성했다. Python 3.6 표준 라이브러리만 사용한다.
"""
import hashlib
import os
import subprocess

MANIFEST_RELATIVE_PATH = os.path.join("configs", "protected_manifest.sha256")
THIRD_PARTY_RELATIVE_PATH = "third_party"


def sha256_of_file(path):
    """파일의 SHA-256 해시를 계산한다.

    입력: path - 해시를 계산할 파일 경로
    반환: 16진수 소문자 문자열(64자)
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        chunk = file.read(1 << 20)
        while chunk:
            hasher.update(chunk)
            chunk = file.read(1 << 20)
    return hasher.hexdigest()


def read_manifest(repo_root):
    """매니페스트를 읽어 {저장소 기준 상대경로: 기대 해시} 사전으로 돌려준다.

    입력: repo_root - 저장소 루트 경로
    반환: dict. 키는 '/' 구분 상대경로(예: third_party/ghoddoosian/Training.py), 값은 SHA-256 문자열
    """
    expected_hashes = {}
    with open(os.path.join(repo_root, MANIFEST_RELATIVE_PATH), "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            digest, relative_path = line.split("  ", 1)
            expected_hashes[relative_path] = digest
    return expected_hashes


def list_third_party_files(repo_root):
    """third_party 아래의 모든 파일(하위 폴더 포함)을 상대경로 목록으로 돌려준다.

    입력: repo_root - 저장소 루트 경로
    반환: '/' 구분 상대경로 문자열의 정렬된 list
    """
    found_paths = []
    top = os.path.join(repo_root, THIRD_PARTY_RELATIVE_PATH)
    for folder, _subfolders, file_names in os.walk(top):
        for name in file_names:
            full_path = os.path.join(folder, name)
            found_paths.append(os.path.relpath(full_path, repo_root).replace("\\", "/"))
    return sorted(found_paths)


def check_manifest(repo_root):
    """vendoring 파일 44개의 해시를 매니페스트와 대조하고, third_party에 새 파일이 없는지 확인한다.

    입력: repo_root - 저장소 루트 경로
    반환: 검사한 파일 수(int). 해시가 다르거나, 파일이 없거나, 목록에 없는 파일(__pycache__ 포함)이 있으면 RuntimeError
    """
    expected_hashes = read_manifest(repo_root)
    problems = []
    for relative_path, expected in sorted(expected_hashes.items()):
        full_path = os.path.join(repo_root, relative_path)
        if not os.path.isfile(full_path):
            problems.append("없음: " + relative_path)
        elif sha256_of_file(full_path) != expected:
            problems.append("해시 불일치: " + relative_path)
    for relative_path in list_third_party_files(repo_root):
        if relative_path not in expected_hashes:
            problems.append("목록에 없는 파일: " + relative_path)
    if problems:
        raise RuntimeError("third_party 무결성 검사 실패:\n  " + "\n  ".join(problems))
    return len(expected_hashes)


def run_git(repo_root, arguments):
    """git 명령을 실행하고 표준출력을 문자열로 돌려준다.

    입력: repo_root - 저장소 루트 경로, arguments - git 인자 list(예: ["rev-parse", "HEAD"])
    반환: 앞뒤 공백을 뺀 출력 문자열. git이 없거나 실패하면 None
    """
    try:
        result = subprocess.run(["git", "-C", repo_root] + arguments, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_git_info(repo_root):
    """현재 커밋 해시와, 추적 중인 파일이 수정된 상태(dirty)인지 알려 준다.

    입력: repo_root - 저장소 루트 경로
    반환: {"commit": 해시 또는 "unknown", "dirty": True/False 또는 None(알 수 없음)}
          추적하지 않는 새 파일(실험 출력 등)은 dirty 판정에 넣지 않는다.
    """
    commit = run_git(repo_root, ["rev-parse", "HEAD"])
    status = run_git(repo_root, ["status", "--porcelain", "--untracked-files=no"])
    dirty = None if status is None else (status != "")
    return {"commit": commit or "unknown", "dirty": dirty}


if __name__ == "__main__":
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("무결성 검사 통과: %d개 파일" % check_manifest(REPO_ROOT))
    print("git 정보: %s" % get_git_info(REPO_ROOT))
