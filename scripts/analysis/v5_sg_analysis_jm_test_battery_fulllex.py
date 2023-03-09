# 3/28 update: do not need to rename the files in between parts
#              drop the silence phoneme
# 4/2 update: added variables at top (to avoid hard coded values)
#             calculating RP is now an option, can get RP or use word activations
# 4/9 update: --- UPDATE FROM V3 ---
#             added class to do command line arguments

# v5_sg_analysis_jm.py
import pickle
import sys

import numpy as np
import pandas
import pandas as pd
import os
import math
import argparse
import time
import psutil
# from csv import reader
# from csv import writer

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
word_tracker = 0
length_effects_DF_list = []
length_effects_param_combo_DF = pd.DataFrame()


# Class to store variables (command line input)
class Analysis:
    def __init__(self, data):
        self.File = data["File"]
        self.alignment = data["alignment"]  # default = post-hoc
        self.specified = data["specified"]  # default = 4
        self.lucek = data["lucek"]  # default = 13
        self.lastSlice = data["lastSlice"]  # default = 12
        self.outfile = data["outfile"]  # default = outfile
        self.respProb = data["respProb"]  # default = FALSE
        self.thresh = 0.41  # default = 0.4
        # self.thresh     = data["thresh"]       # default = 0.4
        # self.topSlices = data["topSlices"]  # default = 50
        self.topSlices = 17151  # default = 50 INCLUDE SILENCE
        # self.words      = data["words"]        # default = 901
        # self.words      = 1162        # default = 901
        self.words = 2  # default = 901
        self.cycles = data["cycles"]  # default = 100
        self.combinations = 3

    ###########################################################
    # PART 1: Calculating response probabilities

    # Numerator value of the Luce Choice Rule
    def numerator(self, val):
        return math.exp(self.lucek * val)

    # Selects one copy from each word
    # input: dataframe of one word in the simulation
    #   Columns: copy of word   Rows: timestep
    # output: returns a list of activation values for the word at each timestep
    def selectWordUnits(self, df):
        # print('selectWordUnits; alignment = ' + str(self.alignment))
        # print('df\n', df.head())
        df = df.drop(
            columns=["Cycle", 'Fullstring', "alpha_if", "alpha_pw", 'alpha_fp', "alpha_wp", "gamma_f", "gamma_p",
                     "gamma_w", "param_combo", "Target", "Word"])
        # df = (df.drop(columns=["Cycle", "Fullstring", "Target", "Feedback", "Noise", "Word"]))
        # print('df after drop\n', df.head())

        # copy = specified
        if self.alignment == "specified":
            # print("specified")
            df = df.loc[:, [self.specified]].squeeze()

        # copy = max activated
        elif self.alignment == "post-hoc":
            # print("post")
            maxValCol = pd.to_numeric(df.stack()).idxmax()[1]
            df = df.loc[:, [maxValCol]].squeeze()

        # copy = max at each timestep (not same copy throughout)
        elif self.alignment == "ad-hoc":
            # print("ad-hoc")
            df = df.max(axis=1)

        # print("select word units\n",df.head())
        return list(df)

    # Calculates the response probability
    # input: dataframe of the activation values 
    # output: dataframe of the response probabilities 
    def calcRespProb(self, df):
        df = df.applymap(self.numerator)
        df = df.div(df.sum(axis=1), axis=0)
        return df

    # input: data(frame) and target
    # output: returns accuracy and RT for one simulation
    def getAccRT(self, data, trg):

        # print(data)
        # test if target exceeds threshold; this is the first
        # condition that must be true for a correct simulation
        if data[trg].max() > self.thresh:
            print("MAX", data[trg].max())
            trg_exceeds = data[data[trg].gt(self.thresh)].index[0]
        else:
            trg_exceeds = np.NaN

        # get the max of the data excluding the target
        others_max = data.max().drop(pd.to_numeric(data.max()).idxmax()).max()

        # now check if that max is greater than the thresh; if so,
        # this will be an error
        if others_max >= self.thresh:
            others_below = 0
        else:
            others_below = 1

        if trg_exceeds > 0 and others_below > 0:
            acc = 1
        else:
            acc = 0

        return [acc, trg_exceeds, data[trg].max(), others_max]

    # Processes a chunk (all data relevant for 1 simulation)
    # One chunk = data for all words, given 1 target
    # Input: dataframe
    # Output: accuracy and RT for this simulation
    def processChunk(self, dataDF):

        # dataDF.head()

        # first we get rid of characters that can be problematic
        # dataDF.Word = dataDF.Word.apply(lambda x: x.replace('-','Q'))
        # dataDF.Word = dataDF.Word.apply(lambda x: x.replace('S','h'))
        # dataDF.Word = dataDF.Word.apply(lambda x: x.replace('^','x'))
        #
        # dataDF.Target = dataDF.Target.apply(lambda x: x.replace('-','Q'))
        # dataDF.Target = dataDF.Target.apply(lambda x: x.replace('S','h'))
        # dataDF.Target = dataDF.Target.apply(lambda x: x.replace('^','x'))

        # WE DONT NEED TO DO RTHAT, RIGHT ?

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(dataDF)
        sys.exit()
        # get the word list
        lexicon = dataDF.Word.unique()
        # print("lexicon", lexicon)

        # define an empty data frame
        outputDF = pd.DataFrame()

        # define a couple variables relevant for progerss reports
        # lexlen = len(lexicon)
        # atword = 0

        # for each word in dataDF select one copy for each word, based on
        # alignment (this happens in selectWordUnits)
        # print(dataDF.head())
        # dataDF = inputDF
        for word in lexicon:
            # atword = atword + 1
            # print('\t\t' + str(atword) + ' of ' + str(lexlen) + '\t' + str(word) + '                    ', end = '\r')
            copy = self.selectWordUnits(dataDF.loc[dataDF["Word"] == word])
            outputDF[word] = copy

        # print(outputDF.head())

        # drop the silence phoneme
        # outputDF = outputDF.drop(columns=["-"])

        # put the target data into its own data frame. We do this because sometimes,
        # especially when noise is high, the target might not make it into the topSlices,
        # which leads to missing key errors
        thetarget = dataDF['Target'].iloc[0]
        # print(thetarget)
        # print(outputDF.keys())
        # print('ABOUT TO GET TARGETDF for TARGET ' + thetarget)
        # print(outputDF)
        targetDF = outputDF[thetarget]
        # print(targetDF.head())
        # print(targetDF.keys())

        # now let's select the topSlices - 1 other items to include
        otherDF = outputDF.drop(columns=[thetarget])
        topIndex = otherDF.max().sort_values(ascending=False).index[:(self.topSlices - 1)]
        otherDF = otherDF[topIndex]

        # now let's concatenate them back together
        evalDF = pd.concat([targetDF, otherDF], axis=1)
        evalDF.columns.values[0] = thetarget
        # print(evalDF)
        # print(evalDF.columns)

        accrt = self.getAccRT(evalDF, thetarget)
        target_peakval = evalDF.iloc[:,0].max()
        target_peaktime = evalDF.iloc[:,0].idxmax()
        print(target_peakval)
        print(target_peaktime)
        print(accrt)

        # length-effects
        global word_tracker
        global length_effects_param_combo_DF
        # take 5 steps of the target word activation at a time
        print(evalDF.iloc[::5,0])
        print(thetarget)
        print(word_tracker)

        print('IF STATEMENT')

        if word_tracker < (self.words - 1):
            length_effects_param_combo_DF.insert(word_tracker, thetarget, evalDF.iloc[::5, 0])
            print(length_effects_param_combo_DF)
            word_tracker = word_tracker + 1
            print(word_tracker)
            if word_tracker == (self.words - 1):
                length_effects_DF_list.append(length_effects_param_combo_DF)
                length_effects_param_combo_DF = pd.DataFrame()
                word_tracker = 0
        #
        # if word_tracker >= (self.words-1):
        #     print('ELSE STATEMENT')
        #     print('THE TARGET', thetarget)
        #     print('WORD TRACKER', word_tracker)
        #     print('DF NOW')
        #     print(length_effects_param_combo_DF)
        #     word_tracker = 0
        #     length_effects_DF_list.append(length_effects_param_combo_DF)
        #     print('DF SHOULD BE NONE')
        #     length_effects_param_combo_DF = pd.DataFrame()
        #     print(length_effects_param_combo_DF)
        #     length_effects_param_combo_DF.insert(word_tracker,thetarget,evalDF.iloc[::5,0])
        #     print(length_effects_param_combo_DF)
        #     word_tracker = word_tracker + 1
        #     print(word_tracker)
        # else:
        #     length_effects_param_combo_DF.insert(word_tracker,thetarget,evalDF.iloc[::5,0])
        #     print(length_effects_param_combo_DF)
        #     word_tracker = word_tracker + 1
        #     print(word_tracker)

        #ok looks good ! now just remember to export the df_list

        print('exit')
        # sys.exit()

        # cohort_peakval = None
        # cohort_peaktime = None
        # rhyme_peakval = None
        # rhyme_peaktime = None
        # unrelated_peakval = None
        # unrelated_peaktime = None

        # if accrt[0] == 1:

        pandas.set_option('display.max_rows', None)
        print(thetarget)
        # print(evalDF)

        # crate dataframe with all words that reach a minimum threshold of -1 (all words do, but you can modify it)
        compDF = evalDF[evalDF.columns[evalDF.max() >= -1]]

        # import rhymes
        with open(rhymes_cohorts_dir + 'fulllex_rhymes_cohorts.pickle', 'rb') as f:
            cohorts_rhymesDF = pickle.load(f)
        # print(cohorts_rhymesDF)

        # cohorts and rhymes
        cohorts_rhymesDF = cohorts_rhymesDF.loc[cohorts_rhymesDF['Phonology']==thetarget]
        # print(cohorts_rhymesDF)

        # add rhymes and cohorts for test purposes
        # cohorts_rhymesDF.at[0,'rhymes'] = ['siks','slEdIG']
        # cohorts_rhymesDF.at[0,'cohorts'] = ['mltrt','xsAsxnet','YtInxrEri']
        # print(cohorts_rhymesDF)

        cohorts = cohorts_rhymesDF['cohorts'].tolist()[0]
        rhymes = cohorts_rhymesDF['rhymes'].tolist()[0]
        # print(cohorts)
        # print(type(cohorts))
        # print(rhymes)

        # get all rhymes and cohorts that have an activation of > -1
        meancohortDF = evalDF.loc[:,evalDF.columns.isin(cohorts)].mean(axis=1)
        meanrhymeDF = evalDF.loc[:,evalDF.columns.isin(rhymes)].mean(axis=1)
        meanunrelatedDF = evalDF.loc[:,~evalDF.columns.isin(cohorts) & ~evalDF.columns.isin(rhymes)].mean(axis=1)
        # print(meanunrelatedDF)
        # print('meancohortdf')
        # print(meancohortDF)

        cohort_peakval = meancohortDF.max()
        cohort_peaktime = meancohortDF.idxmax()
        rhyme_peakval = meanrhymeDF.max()
        rhyme_peaktime = meanrhymeDF.idxmax()
        unrelated_peakval = meanunrelatedDF.max()
        unrelated_peaktime = meanunrelatedDF.idxmax()

        print(cohort_peakval,cohort_peaktime)
        print(rhyme_peakval,rhyme_peaktime)
        print(unrelated_peakval,unrelated_peaktime)



            #if no rhymes are found, do not include them in the average
            # sys.exit()

        # if want RPs
        # can leave that as false
        # softmax mathematics
        if self.respProb:
            evalDF = self.calcRespProb(evalDF)

        # now we get accuracy and RT
        print(accrt)
        accrt = accrt + [cohort_peakval, cohort_peaktime, rhyme_peakval, rhyme_peaktime, unrelated_peakval, unrelated_peaktime]
        print(accrt)
        return accrt

    # Here, we are going to read in an input gzip'd CSV in chunks that are based
    # on lexicon size and cycles per simulation -- simsize tells us how many
    # lines there are per simulation. Then we will append to a data file (we are
    # minimizing memory and possibly sacrificing a little speed with the constant
    # file handling, but if we don't do this and it crashes late, we just lose
    # everything if we haven't written to a file).
    def processFile(self):

        # for progress reports
        start_time = time.time()
        atrow = 0
        atchunk = 0

        # number of rows per simuliation
        simsize = (self.words * self.cycles)

        # define header fields
        resultsHeader = ['alpha_if', 'alpha_pw', 'alpha_fp', 'alpha_wp', "gamma_f", "gamma_p", "gamma_w", "param_combo",
                         "Target", "Recognized", "RT", "max", "others_max", "cohort_peak_val", "cohort_peak_time", "rhyme_peak_val", "rhyme_peak_time", "unrelated_peak_val", "unrelated_peak_time"]
        # inputHeader = ["Cycle", "Fullstring", "Target", "Feedback", "Noise", "Word"] + list(range(1,34))
        # 33, -x -, x, -0.3, 0.5, 0.025, 0.01, x, -0.3000, -0.3000, -0.3000, -0.2784

        # inputHeader = ["Cycle", "Fullstring", "Target", "rest_w", "alpha_if", "alpha_pw", "alpha_wp", "Word"] + list(range(1,34))        inputHeader = ["Cycle", "Fullstring", "Target", "rest_w", "alpha_if", "alpha_pw", "alpha_wp", "Word"] + list(range(1,34))
        inputHeader = ["Cycle", "Fullstring", "Target", "alpha_if", "alpha_pw", 'alpha_fp', "alpha_wp", "gamma_f",
                       "gamma_p", "gamma_w", "param_combo", "Word"] + list(range(1, 34))

        # here we go!
        for achunk in pd.read_csv(self.File, compression='gzip', names=inputHeader,
                                  chunksize=simsize, skipinitialspace=True, header=None):
            # make sure simzise is applicable to your simulation

            # print("chunk\n",achunk)
            # for progresss reports
            chunk_start_time = time.time()

            # reinitialize resultsDF -- we're going to just print 1 line to csv for each chunk
            resultsDF = pd.DataFrame(columns=resultsHeader)
            # print('resultsDF/n', resultsDF.head())

            # initialize inputDF and put the current chunk in there
            inputDF = pd.DataFrame(columns=inputHeader)
            inputDF = inputDF.append(achunk)

            # define the target string
            thetarget = inputDF['Target'].iloc[0]

            # skip silence word
            if thetarget == '-' or thetarget == 'Q':
                continue

            # process this chunk
            simresult = self.processChunk(inputDF)
            # print('simresult\n', simresult)

            # set values for output file
            # rest_w = inputDF['rest_w'].iloc[0]
            alpha_if = inputDF['alpha_if'].iloc[0]
            alpha_pw = inputDF['alpha_pw'].iloc[0]
            alpha_fp = inputDF['alpha_fp'].iloc[0]
            alpha_wp = inputDF['alpha_wp'].iloc[0]
            gamma_f = inputDF['gamma_f'].iloc[0]
            gamma_p = inputDF['gamma_p'].iloc[0]
            gamma_w = inputDF['gamma_w'].iloc[0]
            param_combo = inputDF['param_combo'].iloc[0]
            # # set vals for the things in your header

            # put the results in resultsDF -- always position 0 because we
            # always reinitialize it. Probably a less scruffy way to do this...
            simresultlist = [alpha_if, alpha_pw, alpha_fp, alpha_wp, gamma_f, gamma_p, gamma_w, param_combo,
                             thetarget] + simresult
            # print('simresutllist', simresultlist)
            # print('resultsdf', resultsDF.head())
            resultsDF.loc[0] = simresultlist

            # if we are at the first chunk, we'll use write mode to overwrite existing file
            # (no checking!) and include the header; else, we append
            if atchunk == 0:
                resultsDF.to_csv(self.outfile, mode='w', header=True, na_rep=np.NaN, index=False)
            else:
                resultsDF.to_csv(self.outfile, mode='a', header=False, na_rep=np.NaN, index=False)

            # progress report
            atchunk = atchunk + 1
            chunk_time = str(round((time.time() - chunk_start_time), 2))
            time_per_word = (time.time() - start_time) / atchunk
            minutes_to_go = str(round(((time_per_word * (self.words * self.combinations - atchunk) / 60)), 2))
            time_per_word = str(round(time_per_word, 2))

            # print('self.words')
            # print(self.words)

            print(str(simresultlist) + ': ' + str(atchunk) + ' ' + thetarget + ': ' + chunk_time + ' secs, ' +
                  time_per_word + ' secs/word, ' + str(self.words * self.combinations - atchunk) + ' words to go, ' +
                  str(minutes_to_go) + ' minutes to go')

        # That's it! We are done!
        print('DONE.')


###########################################################

# plotting the timecourse exercise

# can try integrating threshold checking in this file, or
# figure out what's the minimum amonut of data needed to test different threshold combos
# instead get target peak activation, and other peak activation for all words, for each paramter combination
# possibly implement the colmunn that tells you the number of parameter combo (eg. combo #4)
# across the ranges of thresholds from .1 to .99, what will be the accuracy for these ranges as you look into the file

# Creates command line arguments
def parse():
    parser = argparse.ArgumentParser(prog='analysis',
                                     description='Calculates accuracy and reaction time from simulation files')

    parser.add_argument('File', metavar='file', type=str, help='the csv file to process')
    parser.add_argument('-a', '--alignment', action='store', choices=['specified', 'post-hoc', 'ad-hoc'],
                        default='post-hoc',
                        help='type of alignment')
    parser.add_argument('-s', '--specified', action='store', type=int, metavar='COPY', default=4,
                        help='word copy X when alignment is specified')
    parser.add_argument('-k', '--lucek', action='store', type=int, metavar='K', default=13,
                        help='Luce Choice Rule k value')
    parser.add_argument('-ls', '--lastSlice', action='store', metavar='SLICE', type=int, default=12,
                        help='first N-2 slices to find the most activated slice')
    parser.add_argument('-of', '--outfile', type=str, metavar='FILENAME', default='outfile',
                        help='name for output file')
    parser.add_argument('-rp', '--respProb', action='store_true',
                        help='use response probabilities (rather than word activations)')
    parser.add_argument('-th', '--thresh', action='store', type=float, default=0.4, help='accuracy threshold value')
    parser.add_argument('-ts', '--topSlices', action='store', type=int, metavar='SLICE', default=50,
                        help='top N activated slices')
    parser.add_argument('-w', '--words', action='store', type=int, metavar='SIMSIZE', default=901,
                        help='Number of words in lexicon (used to parse file)')
    parser.add_argument('-c', '--cycles', action='store', type=int, metavar='SIMSIZE', default=100,
                        help='Number of cycles per simulation (used to parse file)')

    return parser


if __name__ == "__main__":
    parser = parse()
    args = parser.parse_args()
    a = Analysis(vars(args))

    rhymes_cohorts_dir = '/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/scripts/rhymes-cohorts/fulllex/'
    # beg_time = time.time()

    attributes = vars(a)
    for attr in attributes:
        print(attr, ':', attributes[attr])

    # if user has not specified an outfile, let's make a new name based on the input file;
    # this allows users to run this command with multiple files (separate runs) without
    # having to define the outfile
    if a.outfile == 'outfile':
        a.outfile = str(a.File) + '_results_0.42.csv'

    a.processFile()

    with open('length_effects_DF_list.pkl','wb') as f:
        pickle.dump(length_effects_DF_list,f)


    l1, l2, l3 = psutil.getloadavg()
    CPU_use = (l3 / os.cpu_count()) * 100

    print(CPU_use, 'cpu')

    # print('total time:', time.time() - beg_time)

    # a.respProbCsv()
    # a.filesToAnalysisCsv()
