# Nikita Sossounov 2022
# This script does not account for true word frequency

import pandas as pd

indir = '/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/data/'
outdir = './'

df = pd.read_csv(indir+'lemmalex-with-trace-pronunciations-unique.csv')

fulllex = open(outdir+"fulllex.xml", "w")

fulllex.write('<?xml version="1.0" encoding="UTF-8"?>\n')
fulllex.write("<lexicon xmlns='http://xml.netbeans.org/examples/targetNS'\n")
fulllex.write("xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'\n")
fulllex.write("xsi:schemaLocation='http://xml.netbeans.org/examples/targetNS file:/Ted/develop/WebTrace/Schema/WebTraceSchema.xsd'>\n")

for i in range(0, len(df)):
    fulllex.write("<lexeme><phonology>"+df['Phonology'][i]+"</phonology><frequency>1</frequency></lexeme>\n")
fulllex.write('<lexeme><phonology>-</phonology><frequency>1</frequency></lexeme>')
fulllex.close()

fulllex_code = open('fulllex-code.txt','w')

for i in range(0, len(df)):
    fulllex_code.write("    { phon: '"+df['Phonology'][i]+"', freq: 1, prime: 0 },\n")
fulllex_code.write("    { phon: '-', freq: 1000, prime: 0 },")
fulllex_code.close()

