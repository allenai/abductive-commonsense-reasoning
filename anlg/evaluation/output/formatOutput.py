#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:17:01 2017

@author: kireeti

"""
import sys
file=open(sys.argv[1])
outFile=open(sys.argv[2],'w')
for line in file:
    words=line.split()
    outLine=""
    #print(words)
    for word in words:
        outLine=outLine+word.split("￨")[0]+" "
    #print(outLine)
    outFile.write(outLine+"\n")
file.close()
outFile.close()


    
        