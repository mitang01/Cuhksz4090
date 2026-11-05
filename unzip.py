import py7zr

archive = py7zr.SevenZipFile('0416.7z', mode='r')
archive.extractall(path="/share/workspace2/tangmi")
archive.close()