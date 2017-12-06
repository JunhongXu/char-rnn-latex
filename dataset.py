import tarfile
import gzip
from torch.utils.data import Dataset, DataLoader
import glob
import io
import chardet


dataset_path = '/media/junhong/LargeStorage/arXiv-papers/papers/raw_latex.txt'

######## utility functions #########
def extract_text():
    """
        Extract everything inside basepath to savepath
    """
    basepath = '/media/junhong/LargeStorage/arXiv-papers/papers/2010/'
    paper_folders = glob.glob(basepath+'*')
    content = ''
    findex = 0
    for folder in paper_folders:
        papers = glob.glob(folder + '/*.gz')
        print('Reading folder %s: %i/%i' % (folder, findex, len(paper_folders)))
        npapers = 0
        for paper in papers:
            # extract .tex file from gzip file
            print('\rReading paper %s: %i/%i' %(paper, npapers, len(papers)), end='', flush=True)
            with gzip.open(paper) as gz:
                file_like_obj = io.BytesIO(gz.read())
            try:
                with tarfile.open(fileobj=file_like_obj) as tar:
                    for member in tar.getmembers():
                        name = member.name
                        if name.endswith('.tex'):
                            tex = tar.extractfile(member)
                            content += tex.read().decode('utf-8')
            except:
                # a regular file
                encoding = chardet.detect(file_like_obj.read())
                encoding = encoding['encoding']
                if encoding is not None:
                    content += file_like_obj.read().decode(encoding)

            file_like_obj.close()
            npapers += 1
        print('\n')
        findex += 1
    with open(dataset_path, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    extract_text()