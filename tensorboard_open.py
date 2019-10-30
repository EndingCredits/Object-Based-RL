import re
import subprocess
import sys
import webbrowser

p = re.compile("http://CSPC416b:\d+")
cmd = "tensorboard --port 0 --logdir " + sys.argv[1]

proc = subprocess.Popen(cmd,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                        universal_newlines = True,
                        shell = True)
                        
while proc.poll() is None:
    line = proc.stderr.readline()
    if line != '':
        print(line, end='')
        url = p.findall(line)
        if url:
            webbrowser.open(url[0], autoraise=False)
        
#for line in sys.stdin:
#    print(line)
#    url = p.findall(line)
#    if url:
#        webbrowser.open_new_tab(url[0])

