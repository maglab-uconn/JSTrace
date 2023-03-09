# Nikita Sossounov 2022
# This script does not account for true word frequency

import pandas as pd

indir = '/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/scripts/pseudo-lexicon/'
outdir = './'

df = pd.read_csv(indir+'pseudo-lexicon.csv')

tlex = open(outdir+"tlex.xml", "w")

tlex.write('<?xml version="1.0" encoding="UTF-8"?>\n')
tlex.write("<lexicon xmlns='http://xml.netbeans.org/examples/targetNS'\n")
tlex.write("xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'\n")
tlex.write("xsi:schemaLocation='http://xml.netbeans.org/examples/targetNS file:/Ted/develop/WebTrace/Schema/WebTraceSchema.xsd'>\n")

for i in range(0, len(df)):
    tlex.write("<lexeme><phonology>"+df['Phonology'][i]+"</phonology><frequency>1</frequency></lexeme>\n")
tlex.write('<lexeme><phonology>-</phonology><frequency>1</frequency></lexeme>')
tlex.close()

tlex_code = open('tlex-code.txt','w')

for i in range(0, len(df)):
    tlex_code.write("    { phon: '"+df['Phonology'][i]+"', freq: 1, prime: 0 },\n")
tlex_code.write("    { phon: '-', freq: 1000, prime: 0 },")
tlex_code.close()

