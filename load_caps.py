import sys
import re
import json
import numpy as np

if __name__ == "__main__":
    filename = sys.argv[1]
    np.random.seed(0)
    input_ext = ".json"

    def clean_newline(text):
        return re.sub('\n+', " ", text).strip()

    caps = {}
    print("Building dictionary...")
    with open(filename+input_ext, 'r', encoding='utf8') as f:
        text = f.read()
        anno_start = text.index("\"annotations\"")
        text = text[anno_start:]
        text = text[15:-1] 
        jcap = json.loads(text)
        for i in range(len(jcap)):
            image_id = jcap[i]['image_id']
            caption = jcap[i]['caption']
            if image_id in caps:
                caps[image_id].append(caption)
            else:
                caps[image_id] = [caption]

    print("Writing dictionary to files...")
    src_file = open(filename+".src.en", 'w')
    ref_files = {}
    for i in range(4):
        ref_files[i] = open(filename+".ref"+str(i)+".en", 'w')

    for key in caps.keys():
        idxs = np.random.permutation(5)
        src_file.write(clean_newline(caps[key][idxs[0]]) + '\n')
        for i in range(4):
            ref_files[i].write(clean_newline(caps[key][idxs[i+1]]) + '\n')

    src_file.close()
    for i in range(4):
        ref_files[i].close()