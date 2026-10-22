"""Platform-local configuration paths and fail-closed credential storage."""
from __future__ import annotations
import os
import stat
import subprocess
import tempfile
from pathlib import Path


def config_directory():
    if os.name == 'nt':
        return Path(os.environ['LOCALAPPDATA']) / 'Betalens/DBManager/config'
    return Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home()/'.config'))) / 'betalens/db-manager'


def state_directory():
    if os.name == 'nt':
        return Path(os.environ['LOCALAPPDATA']) / 'Betalens/DBManager/state'
    return Path(os.environ.get('XDG_DATA_HOME', str(Path.home()/'.local/share'))) / 'betalens-db-manager'


def private_path(path: Path):
    path = path.expanduser().resolve()
    for parent in (path.parent, *path.parent.parents):
        if (parent/'.git').exists():
            raise ValueError('私有配置必须位于 Git 工作树之外，包括被忽略的目录')
    if os.name != 'nt' and 'microsoft' in __import__('platform').release().lower():
        # WSL credentials must remain on the Linux filesystem, including symlinks.
        result = subprocess.run(['stat','-f','-c','%T',str(next(p for p in (path.parent,*path.parent.parents) if p.exists()))],capture_output=True,text=True,check=True)
        if result.stdout.strip() in {'9p','drvfs','fuseblk','ntfs','smb2','cifs'}:
            raise ValueError('WSL 私密配置必须保存在 Linux 文件系统')
    return path


def restrict(path: Path, directory=False):
    if os.name == 'nt':
        user = subprocess.run(['whoami'],capture_output=True,text=True,check=True).stdout.strip()
        suffix = '(OI)(CI)' if directory else ''
        subprocess.run(['icacls',str(path),'/inheritance:r','/grant:r',f'{user}:{suffix}F',f'SYSTEM:{suffix}F'],capture_output=True,check=True)
    else:
        path.chmod(0o700 if directory else 0o600)
        if stat.S_IMODE(path.stat().st_mode) != (0o700 if directory else 0o600):
            raise ValueError('无法设置私有配置权限')


def atomic_private(path: Path, content: str, *, credential=False):
    path = private_path(path) if credential else path.expanduser().resolve()
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    restrict(path.parent,True)
    fd, temporary = tempfile.mkstemp(prefix='.db-manager-',dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            restrict(temporary)
            f.write(content);f.flush();os.fsync(f.fileno())
        if credential: private_path(path)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
