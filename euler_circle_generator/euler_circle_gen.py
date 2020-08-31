# first version coded by Duo Wang (wd263@cam.ac.uk) 2018
# slightly modified by Tiansi Dong (dongt@bit.uni-bonn.de) 2020

from matplotlib import pyplot as plt
from matplotlib_venn import venn2,venn2_circles
import random
import glob
import os
import pickle
import numpy as np
from euler_circle_lib_duo import *

num_each_class=500

'''
Relational statement format:
Letter:A-Z for different entity
Relationship for Venn-2: 
1. A contains B ['A','>','B']               P(B,A)............ all B are A
2. A contained in B ['A','<','B']           P_bar(A,B) ....... all A are B
3. A intersects B ['A','&','B']             O(A,B)............ some A are B, some B are A, some A is not B, some B is not A
4. A does not intersect B ['A','!','B']     D(A,B)............ no A is B, no B is A
'''

relational_operator=['>','<','&','!']


def area_gen(operator):
    if operator=='>':
        Ab=100
        aB=0
        AB=random.uniform(10,Ab)
        return (Ab,aB,AB)
    elif operator=='<':
        Ab=0
        aB=100
        AB=random.uniform(10,aB)
        return (Ab,aB,AB)
    elif operator=='&':
        Ab=random.uniform(10,100)
        aB=random.uniform(10,100)
        AB=200-Ab-aB
        return (Ab,aB,AB)
    elif operator=='!':
        Ab=random.uniform(10,100)
        aB=random.uniform(10,100)
        AB=0
        return (Ab,aB,AB)
    else:
        print('invalid relational operator')


def inference(op1,op2):
    if op1=='>':                    # all B are A
        if op2=='>':                    # all C are B
            op3=['>']                       # all C are A
        if op2=='<':                    # all B are C
            op3=['<','>','&']               # not [A disconnects from C]
        if op2=='&':                    # O(B,C)
            op3=['&','>']                   # O(A,C) or P(C, A)
        if op2=='!':                    # D(B,C)
            op3=['>','!','&']               # not [C contains A]
    elif op1=='<':                  # A contained in B
        if op2=='>':                    # B contains C
            op3=['>','<','&']               # all relations are possible? op3=['>','<','&', '!']
        if op2=='<':                    # B contained in C
            op3=['<']                       # A contained in C
        if op2=='&':                    # B partially overlaps with C
            op3=['&','<','!']               # not [A contains C]
        if op2=='!':                    # B disconnects from C
            op3=['!']                       # A disconnects from C
    elif op1=='&':                  # A partially overlaps B
        if op2=='>':                    # B constains C
            op3=['>','&','!']               # not [A contained in C]
        if op2=='<':                    # B contained in C
            op3=['<','&']                   # A contained in C or A overlaps with C
        if op2=='&':                    # B partially overlaps with C
            op3=['&','>','<','!']           # all relations are possible? op3=['>','<','&', '!']
        if op2=='!':                    # B disconnects from C
            op3=['>','!','&']               # not [A contained in C]
    elif op1=='!':                  # A disconnects from B
        if op2=='>':                    # B contains C
            op3=['!']                       # A disconnects from C
        if op2=='<':                    # B contained in C
            op3=['<','!','&']               # not [A contains C]
        if op2=='&':                    # B partially overlaps with C
            op3=['&','!','<']               # not [A contains C]
        if op2=='!':                    # B disconnects from C
            op3=['>','!','&','<']           # all relations are possible? op3=['>','<','&', '!']
    label=np.zeros(4)
    for op in op3:
        label[relational_operator.index(op)]=1
    return op3,label


def generate_euler_circles(ipath="", opath=''):
    diag_dict={}
    img_counter=0
    euler_record=np.zeros((4*4*num_each_class,4),dtype=np.uint8)
    if "Bamalip" in ipath:
        # all relation.n.01 abstraction.n.06, all possession.n.02 relation.n.01: some abstraction.n.06 possession.n.02; no abstraction.n.06 possession.n.02
        a_relational_operator = ['<']
        b_relational_operator = ['<']
        op3 = ['<','>','&']
    elif "Barbara" in ipath:
        # all space.n.01 attribute.n.02, all attribute.n.02 entity.n.01: all space.n.01 entity.n.01; some-not space.n.01 entity.n.01
        a_relational_operator = ['<']
        b_relational_operator = ['<']
        op3 = ['<']
    elif "Barbari" in ipath:
        # all medium.n.07 substance.n.01, all substance.n.01 entity.n.01: some medium.n.07 entity.n.01; no medium.n.07 entity.n.01
        a_relational_operator = ['<']
        b_relational_operator = ['<']
        op3 = ['<','>','&']
    elif "Baroco" in ipath:
        # some-not layer.n.01 device.n.01, all brass.n.02 device.n.01: some-not layer.n.01 brass.n.02; all layer.n.01 brass.n.02
        a_relational_operator = ['>', '!', '&']
        b_relational_operator = ['>']
        op3 = ['>', '!', '&']
    elif "Bocardo" in ipath:
        # all disgust.n.01 part.n.02, some-not disgust.n.01 dislike.n.02: some-not part.n.02 dislike.n.02; all part.n.02 dislike.n.02
        a_relational_operator = ['>']
        b_relational_operator = ['>', '!', '&']
        op3 = ['>', '!', '&']

    elif "Calemes_Camestres" in ipath:
        # no public_relations_person.n.01 creator.n.02, all artist.n.01 creator.n.02: no public_relations_person.n.01 artist.n.01; some public_relations_person.n.01 artist.n.01
        a_relational_operator = ['!']
        b_relational_operator = ['>']
        op3 = ['!']
    elif "Camestros_Calemos" in ipath:
        # no backwater.n.02 municipality.n.01, all city.n.01 municipality.n.01: some-not backwater.n.02 city.n.01; all backwater.n.02 city.n.01
        a_relational_operator = ['!']
        b_relational_operator = ['>']
        op3 = ['>','!','&']
    elif "Celarent_Cesare" in ipath:
        # all whole.n.02 object.n.01, no object.n.01 substance.n.04: no whole.n.02 substance.n.04; some whole.n.02 substance.n.04
        a_relational_operator = ['<']
        b_relational_operator = ['!']
        op3 = ['!']
    elif "Cesaro_Celaront" in ipath:
        # all wave.n.01 movement.n.03, no ending.n.04 movement.n.03: some-not wave.n.01 ending.n.04; all wave.n.01 ending.n.04
        a_relational_operator = ['<']
        b_relational_operator = ['!']
        op3 = ['>', '!', '&']

    elif "Darapti" in ipath:
        # all location.n.01 object.n.01, all location.n.01 physical_entity.n.01: some object.n.01 physical_entity.n.01; no object.n.01 physical_entity.n.01
        a_relational_operator = ['>']
        b_relational_operator = ['<']
        op3 = ['>', '<', '&']
    elif "Darii_Datisi" in ipath:
        # some group.n.01 egyptian.n.01, all egyptian.n.01 african.n.01: some group.n.01 african.n.01; no group.n.01 african.n.01
        a_relational_operator = ['>', '<', '&']
        b_relational_operator = ['<']
        op3 = ['>', '<', '&']
    elif "Disamis_Dimatis" in ipath:
        # all material.n.01 substance.n.01, some material.n.01 covering.n.01: some substance.n.01 covering.n.01; no substance.n.01 covering.n.01
        a_relational_operator = ['>']
        b_relational_operator = ['>', '<', '&']
        op3 = ['>', '<', '&']
    elif "Felapton_Fesapo" in ipath:
        # all noise.n.03 psychological_feature.n.01, no noise.n.03 convulsion.n.04: some-not psychological_feature.n.01 convulsion.n.04; all psychological_feature.n.01 convulsion.n.04
        a_relational_operator = ['>']
        b_relational_operator = ['!']
        op3 = ['>', '!', '&']
    elif "Ferio_Festino_Ferison_Fresison" in ipath:
        # some utterance.n.01 profanity.n.01, no utterance.n.01 quoter.n.01: some-not profanity.n.01 quoter.n.01; all profanity.n.01 quoter.n.01
        a_relational_operator = ['>', '<', '&']
        b_relational_operator = ['!']
        op3 = ['>', '!', '&']

    label = np.zeros(4)
    for op in op3:
        label[relational_operator.index(op)] = 1

    for i in a_relational_operator:
        for j in b_relational_operator:
            for t in range(num_each_class):
                print(i,j)
                diag_id=i+j+'_'+str(t)
                file_prefix=opath+'/'+'premise_'+diag_id
                op_num_1=gen_circle(i,file_prefix+'_1.jpg','r','g')
                op_num_2=gen_circle(j,file_prefix+'_2.jpg','g','b')
                euler_record[img_counter,0]=op_num_1
                euler_record[img_counter,1]=op_num_2
                diag_dict[diag_id]=label
                img_counter+=1


    np.savetxt(opath+'/'+'euler_record.csv',euler_record.astype(np.uint8),delimiter=',')
    with open(opath+'/'+'diag_dict.pickle', 'wb') as fp:
                    pickle.dump(diag_dict, fp)


if __name__ == "__main__":
    inFiles = 'Syllogism/*.txt'
    opath = 'generated_diagram'
    for ipath in glob.glob(inFiles):
        SyllogismModus = ipath[10:-4]
        opath0 = os.path.join(opath, SyllogismModus)
        if not os.path.exists(opath0):
            os.mkdir(opath0)
        generate_euler_circles(ipath=ipath, opath=opath0)

