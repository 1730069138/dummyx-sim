"""Content-addressed, byte-identical immutable model snapshots for saved-state replay."""
from pathlib import Path
import hashlib,os,tempfile,errno,shutil

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def share_model_file(path,cache):
    path=Path(path);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    digest=sha256_file(path);canonical=cache/(digest+'.mjb');size=path.stat().st_size
    try:os.link(path,canonical)
    except FileExistsError:
        if sha256_file(canonical)!=digest:raise RuntimeError('Corrupt model cache: '+str(canonical))
    except OSError as exc:
        if exc.errno!=errno.EXDEV:raise
        with canonical.open('xb') as dst,path.open('rb') as src:shutil.copyfileobj(src,dst)
    os.chmod(canonical,0o444)
    if os.path.samefile(path,canonical):return dict(sha256=digest,cache=str(canonical.resolve()),saved_bytes=0)
    old_last_link=path.stat().st_nlink==1
    handle,temp=tempfile.mkstemp(prefix='.model-link-',dir=path.parent);os.close(handle);os.unlink(temp)
    try:
        try:os.link(canonical,temp)
        except OSError as exc:
            if exc.errno!=errno.EXDEV:raise
            shutil.copyfile(canonical,temp)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    if sha256_file(path)!=digest:raise RuntimeError('Model snapshot changed during sharing')
    shared=os.path.samefile(path,canonical)
    return dict(sha256=digest,cache=str(canonical.resolve()),saved_bytes=size if old_last_link and shared else 0)

def save_model_snapshot(model,destination,cache):
    import mujoco
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    handle,temp=tempfile.mkstemp(prefix='.model-write-',suffix='.mjb',dir=cache);os.close(handle)
    try:
        mujoco.mj_saveModel(model,temp,None);receipt=share_model_file(temp,cache)
        canonical=Path(receipt['cache'])
        try:os.link(canonical,destination)
        except OSError as exc:
            if exc.errno!=errno.EXDEV:raise
            shutil.copyfile(canonical,destination)
        if sha256_file(destination)!=receipt['sha256']:raise RuntimeError('Model save hash mismatch')
        return receipt
    finally:
        if os.path.exists(temp):os.unlink(temp)
