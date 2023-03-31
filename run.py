from utils import LotCorrector

if __name__=="__main__":
    product = 'AniCheck-bIgG'
    lotnum = 23001
    fn_sample_result = "./23001_diff2.csv"

    lc = LotCorrector(fn_sample_result, product, lotnum, degree=6, exp_date = "231222")