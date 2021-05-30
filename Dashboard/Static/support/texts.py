
BM_txt = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/support/BM.txt'
DNN_txt = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/support/DNN.txt'
DTR_txt = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/support/DTR.txt'
EN_txt = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/support/EN.txt'
GBR_txt = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/support/GBR.txt'
KNR_txt = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/support/KNR.txt'
Lasso_txt = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/support/Lasso.txt'
LR_txt = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/support/LR.txt'
RFR_txt = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/support/RFR.txt'
Sequential_txt = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/support/Sequential.txt'
SVM_txt = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/support/SVM.txt'
FD_txt = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/support/First_Description.txt'


BM_txt_markdown = "\t"
with open(BM_txt) as this_file:
    for a in this_file.read():
        if "\n" in a:
            BM_txt_markdown += "\n \t"
        else:
            BM_txt_markdown += a

DNN_txt_markdown = "\t"
with open(DNN_txt) as this_file:
    for a in this_file.read():
        if "\n" in a:
            DNN_txt_markdown += "\n \t"
        else:
            DNN_txt_markdown += a

DTR_txt_markdown = "\t"
with open(DTR_txt) as this_file:
    for a in this_file.read():
        if "\n" in a:
            DTR_txt_markdown += "\n \t"
        else:
            DTR_txt_markdown+= a

EN_txt_markdown = "\t"
with open(EN_txt) as this_file:
    for a in this_file.read():
        if "\n" in a:
            EN_txt_markdown += "\n \t"
        else:
            EN_txt_markdown += a


GBR_txt_markdown = "\t"
with open(GBR_txt) as this_file:
    for a in this_file.read():
        if "\n" in a:
            GBR_txt_markdown += "\n \t"
        else:
            GBR_txt_markdown += a



KNR_txt_markdown = "\t"
with open(KNR_txt) as this_file:
    for a in this_file.read():
        if "\n" in a:
            KNR_txt_markdown += "\n \t"
        else:
            KNR_txt_markdown += a


Lasso_txt_markdown = "\t"
with open(Lasso_txt) as this_file:
    for a in this_file.read():
        if "\n" in a:
            Lasso_txt_markdown += "\n \t"
        else:
            Lasso_txt_markdown += a


LR_txt_markdown = "\t"
with open(LR_txt) as this_file:
    for a in this_file.read():
        if "\n" in a:
            LR_txt_markdown  += "\n \t"
        else:
            LR_txt_markdown  += a

RFR_txt_markdown = "\t"
with open(RFR_txt) as this_file:
    for a in this_file.read():
        if "\n" in a:
            RFR_txt_markdown += "\n \t"
        else:
            RFR_txt_markdown += a

Sequential_txt_markdown = "\t"
with open(Sequential_txt) as this_file:
    for a in this_file.read():
        if "\n" in a:
            Sequential_txt_markdown += "\n \t"
        else:
            Sequential_txt_markdown += a

SVM_txt_markdown = "\t"
with open(SVM_txt) as this_file:
    for a in this_file.read():
        if "\n" in a:
            SVM_txt_markdown += "\n \t"
        else:
            SVM_txt_markdown += a

First_desc_markdown = "\t"
with open(FD_txt) as this_file:
    for a in this_file.read():
        if "\n" in a:
            First_desc_markdown += "\n \t"
        else:
            First_desc_markdown += a
