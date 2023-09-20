#!/bin/python3

import os, sys, getopt
import subprocess
import glob
from termcolor import colored

#################################################################
#################################################################
def adaptFiles(input_file_names):

    for input_file_name in input_file_names:
        
        if not os.path.exists(input_file_name):
            print(colored("File:","red"),input_file_name, colored("does not exist","red"))
            continue
        output_file_name = "studentsVersions/"+input_file_name
        output_file = open(output_file_name, "w")
        result = subprocess.run(["awk", " /#BEGIN_SOLUTION/{p=1}/#END_SOLUTION/{p=0;print \"    \\\"...rozwiązanie...\\\\n\\\", \";next}!p", input_file_name],
                                text=True, stdout=output_file)
                            
#################################################################
#################################################################
if __name__ == "__main__":

    opts, args = getopt.getopt(sys.argv[1:],"i:",["input="])
    
    for opt, arg in opts:
      if opt in ("-i", "--input"):
         input_file_list = arg.split(",")
    
    print('Input file list: ', input_file_list)
    adaptFiles(input_file_list)
    
     
#################################################################
#################################################################