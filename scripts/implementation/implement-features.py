# Nikita Sossounov 2022
import sys
import pandas as pd

# with open('alph.txt') as f:
#     r = f.read()

def inv(num):
    if num == 0:
        return 1
    else:
        return 0

indir = '/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/data/'
outdir = './'

df = pd.read_csv(indir+'jstrace-phonetic-features-2022-v3.csv')
print(df.columns)

x = open("implemented-features-v3.txt", "w")

print(df)
for i in range(0, len(df)):
    print(i)
    print(df['CODE'][i])
    if df['CODE'][i] == "SKIP" or df['CODE'][i] == "SKIP/x":
        continue

    x.write("    {\n")
    x.write("label: '" + df['CODE'][i] + "',\n")
    x.write("features: [\n")

    x.write("   " + str(df['UNVOICE'][i])+", /*UNVOICE*/\n")
    x.write("   " + str(df['VOICE'][i])+", /*VOICE*/\n")

    x.write("   " + str(df['SONORANT'][i])+", /*SONORANT*/\n")
    x.write("   " + str(df['NONSON'][i])+", /*NONSONORANT*/\n")
    x.write("   " + str(df['LIQUID'][i])+", /*LIQUID*/\n")
    x.write("   " + str(df['SEMI-GLID'][i])+", /*SEMI-GLIDE*/\n")
    x.write("   " + str(df['VOCALIC'][i])+", /*VOCALIC*/\n")

    x.write("   " + str(df['STOP'][i])+", /*STOP*/\n")
    x.write("   " + str(df['NASAL'][i])+", /*NASAL*/\n")
    x.write("   " + str(df['FRICATIVE'][i])+", /*FRICATIVE*/\n")
    x.write("   " + str(df['AFFRICATE'][i])+", /*AFFRICATE*/\n")
    x.write("   " + str(df['APPROXIMANT'][i])+", /*APPROXIMANT*/\n")
    x.write("   " + str(df['LATERAL'][i])+", /*LATERAL*/\n")
    x.write("   " + str(df['BILABIAL'][i])+", /*BILABIAL*/\n")
    x.write("   " + str(df['LABIODENTAL'][i])+", /*LABIODENTAL*/\n")
    x.write("   " + str(df['DENTAL'][i])+", /*DENTAL*/\n")
    x.write("   " + str(df['ALVEOLAR'][i])+", /*ALVEROLAR*/\n")
    x.write("   " + str(df['POSTALVEOLAR'][i])+", /*POSTALVEOLAR*/\n")
    x.write("   " + str(df['PALATAL'][i])+", /*PALATAL*/\n")
    x.write("   " + str(df['VELAR'][i])+", /*VELAR*/\n")
    x.write("   " + str(df['GLOTTAL'][i])+", /*GLOTTAL*/\n")

    x.write("   " + str(df['FB1A'][i])+", /*FB1A*/\n")
    x.write("   " + str(df['FB2A'][i])+", /*FB2A*/\n")
    x.write("   " + str(df['FB3A'][i])+", /*FB3A*/\n")
    x.write("   " + str(df['FB4A'][i])+", /*FB4A*/\n")
    x.write("   " + str(df['FB5A'][i])+", /*FB5A*/\n")

    x.write("   " + str(df['HL1A'][i])+", /*HL1A*/\n")
    x.write("   " + str(df['HL2A'][i])+", /*HL2A*/\n")
    x.write("   " + str(df['HL3A'][i])+", /*HL3A*/\n")
    x.write("   " + str(df['HL4A'][i])+", /*HL4A*/\n")
    x.write("   " + str(df['HL5A'][i])+", /*HL5A*/\n")
    x.write("   " + str(df['HL6A'][i])+", /*HL6A*/\n")

    x.write("   " + str(df['FB1B'][i])+", /*FB1B*/\n")
    x.write("   " + str(df['FB2B'][i])+", /*FB2B*/\n")
    x.write("   " + str(df['FB3B'][i])+", /*FB3B*/\n")
    x.write("   " + str(df['FB4B'][i])+", /*FB4B*/\n")
    x.write("   " + str(df['FB5B'][i])+", /*FB5B*/\n")

    x.write("   " + str(df['HL1B'][i])+", /*HL1B*/\n")
    x.write("   " + str(df['HL2B'][i])+", /*HL2B*/\n")
    x.write("   " + str(df['HL3B'][i])+", /*HL3B*/\n")
    x.write("   " + str(df['HL4B'][i])+", /*HL4B*/\n")
    x.write("   " + str(df['HL5B'][i])+", /*HL5B*/\n")
    x.write("   " + str(df['HL6B'][i])+", /*HL6B*/\n")

    x.write("   " + str(df['SIL1'][i])+", /*SIL1*/\n")
    x.write("   " + str(df['SIL2'][i])+", /*SIL2*/\n")
    x.write("   " + str(df['SIL3'][i])+", /*SIL3*/\n")
    x.write("   " + str(df['SIL4'][i])+", /*SIL4*/\n")


    x.write("],\n")
    x.write("durationScalar: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\n")
    x.write("phonologicalRole: TracePhoneRole.NORMAL,\n")
    x.write("    },\n\n")

x.close()
print('done')
