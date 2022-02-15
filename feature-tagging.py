import os.path
import math
import numpy as np

dirname = os.getcwd() #現在のファイルがあるフォルダのパスを取得

emit = {} #emit保存用
transition = {} #transition保存用
context = {} #context保存用
possible_tags = {} #品詞ラベルの種類を保存

w = {} #重みパラメータ保存用

def train_POS(train_file): #transitionとemitのパラメータを計算する関数
    #引数: (train_file: trainファイル名)
    with open(train_file, mode='rt') as train_f:
        for line in train_f:
            previous = "<s>"
            possible_tags[previous] = 1
            context[previous] = context[previous] + 1 if previous in context else 1
            for wordtag in line.split():
                word, tag = wordtag.split('_')
                transition[previous + " " + tag] = transition[previous + " " + tag] + 1 if (previous + " " + tag) in transition else 1 
                context[tag] = context[tag] + 1 if tag in context else 1
                emit[tag + " " + word] = emit[tag + " " + word] + 1 if (tag + " " + word) in emit else 1
                previous = tag
                possible_tags[previous] = 1
            transition[previous + " </s>"] = transition[previous + " </s>"] + 1 if (previous + "</s>") in transition else 1 

            #transitionの場合: "一つ前の品詞ラベル　現在の品詞ラベル"
            #emitの場合: "品詞ラベル　単語"

        '''
        for key, value in transition.items():
            previous, word = key.split()
            save_list.append("T " + key + " " + str(value / context[previous]))
        
        for key, value in emit.items():
            tag, word = key.split()
            save_list.append("E " + key + " " + str(value / context[tag]))

        np.savetxt(model_file, save_list, fmt="%s")
        load_model(model_file)
        '''
        train_feature(train_file)

'''
def load_model(model_file): #保存したパラメータをロードする関数
    with open(model_file, mode='rt') as model_f:
        for line in model_f:
            wordtype, context, word, prob = line.split()
            #transitionの場合: "T", "一つ前の品詞ラベル", "現在の品詞ラベル", "確率"
            #emitの場合: "E", "品詞ラベル", "単語", "確率"
            possible_tags[context] = 1 #品詞ラベルの種類を保存
            if wordtype == "T":
                vitavi_transition[context + " " + word] = float(prob)
            else:
                vitavi_emit[context + " " + word] = float(prob)
'''

def train_feature(train_file): #重み学習用関数
    #引数: (train_file: trainファイル名)
    with open(train_file, mode='rt') as train_f:
        for line in train_f:
            X = []
            Y_prime = []

            for wordtags in line.split():
                word, tag = wordtags.split('_')
                X.append(word)
                Y_prime.append(tag)

            Y_hat = HMM_viterbi(w, X)
            phi_prime = create_feature(X, Y_prime)
            phi_hat = create_feature(X, Y_hat)
            #print(f'phi_prime: {phi_prime}')
            #print(f'phi_hat: {phi_hat}')
            for prime_key, prime_value in phi_prime.items():
                w[prime_key] = w[prime_key] + prime_value if prime_key in w else prime_value

            for hat_key, hat_value in phi_hat.items():
                w[hat_key] = w[hat_key] - hat_value if hat_key in w else hat_value
            #w += phi_prime - phi_hat
            #print(f'w{X}: {w}')

def HMM_viterbi(w, word): #ビタビアルゴリズム
    #引数: (w: 重みパラメータ、word: 単語列)
    best_score = {}
    best_edge = {}
    best_score["0 <s>"] = 0
    best_edge["0 <s>"] = None

    phi_vitervi = {}

    for i in range(len(word)):#range(len(words) - 1):
        for prev_tag in possible_tags.keys():
            for next_tag in possible_tags.keys():
                w_trans = 0
                w_emit = 0
                if (str(i) + " " + prev_tag) in best_score and (prev_tag + " " + next_tag) in transition:
                    
                    for trans_key, trans_value in create_trans(prev_tag, next_tag).items(): #transition特徴量計算
                        phi_vitervi[trans_key] = phi_vitervi[trans_key] + trans_value if trans_key in phi_vitervi else trans_value
                        w_trans += w[trans_key] * phi_vitervi[trans_key] if trans_key in w else 0
                        
                    for emit_key, emit_value in create_emit(next_tag, word[i]).items(): #emit特徴量計算
                        phi_vitervi[emit_key] = phi_vitervi[emit_key] + emit_value if emit_key in phi_vitervi else emit_value
                        w_emit += w[emit_key] * phi_vitervi[emit_key] if emit_key in w else 0
                        
                    score = best_score[str(i) + " " + prev_tag] + w_trans + w_emit #score計算

                    if (str(i+1) + " " + next_tag) not in best_score or ((str(i+1) + " " + next_tag) in best_score and best_score[str(i+1) + " " + next_tag] < score): #best_score更新
                        best_score[str(i+1) + " " + next_tag] = score
                        best_edge[str(i+1) + " " + next_tag] = str(i) + " " + prev_tag

    best_score[str(len(word)+1) + " </s>"] = 100000000000000000
    best_edge[str(len(word)+1) + " </s>"] = None
    for last_prev_tag in possible_tags.keys():
        if (str(len(word)) + " " + last_prev_tag) in best_score and last_prev_tag + " </s>" in transition:             
            score = best_score[str(len(word)) + " " + last_prev_tag] + - math.log(transition[last_prev_tag + " </s>"])                        
            if best_score[str(len(word)+1) + " </s>"] > score:
                best_score[str(len(word)+1) + " </s>"] = score
                best_edge[str(len(word)+1) + " </s>"] = str(len(word)) + " " + last_prev_tag

    #品詞ラベル列取り出し
    tags = []
    next_edge = best_edge[str(len(word)+1) + " </s>"]
    while next_edge != "0 <s>":
        position, tag = next_edge.split()
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()

    #print(f'join(tags): {" ".join(tags).split()}')

    return ' '.join(tags).split()

def create_feature(X, Y): #特徴量計算
    phi = {}
    for i in range(len(Y)+1):
        first_tag = "<s>" if i == 0 else Y[i-1]
        next_tag = "</s>" if i == len(Y) else Y[i]
        for trans_key, trans_value in create_trans(first_tag, next_tag).items():
            phi[trans_key] = phi[trans_key] + trans_value if trans_key in phi else trans_value

    for i in range(len(Y)):
        for emit_key, emit_value in create_emit(Y[i], X[i]).items():
            phi[emit_key] = phi[emit_key] + emit_value if emit_key in phi else emit_value
    return phi

def create_trans(prev_tag, next_tag): #transitionに関する特徴量計算
    create_t = {}
    create_t["T " + prev_tag + " " + next_tag] = create_t["T " + prev_tag + " " + next_tag] + 1 if ("T " + prev_tag + " " + next_tag) in create_t else 1
    return create_t

def create_emit(next_tag, word): #emitに関する特徴量計算
    create_e = {}
    create_e["E " + next_tag + " " + word] = create_e["E " + next_tag + " " + word] + 1 if ("E " + next_tag + " " + word) in create_e else 1
    if word.endswith("ed"): #過去形であれば特徴量作成
        create_e["SUFED " + next_tag] = create_e["SUFED " + next_tag] + 1 if ("SUFED " + next_tag) in create_e else 1
    if word.endswith("ing"): #現在進行形であれば特徴量作成
        create_e["SUFING " + next_tag] = create_e["SUFING " + next_tag] + 1 if ("SUFING " + next_tag) in create_e else 1
    #if word.istitle():
    #    create_e["CAP " + next_tag] = create_e["CAP " + next_tag] + 1 if ("CAP " + next_tag) in create_e else 1
    create_e["UNI " + word] = create_e["UNI " + word] + 1 if ("UNI " + word) in create_e else 1 #unigramに基づいた特徴量作成
    return create_e

def test_POS(test_file):
    with open(test_file, mode='rt') as test_f:
        Y_prime_test = []
        Y_hat_test = []
        for line in test_f:
            X_test = []
            list_line = []

            for wordtags in line.split():
                word, tag = wordtags.split('_')
                X_test.append(word)
                list_line.append(tag)
                
            Y_prime_test.append(list_line) #正解品詞ラベル追加
            Y_hat_test.append(HMM_viterbi(w, X_test)) #予測結果の品詞ラベル追加

        accuracy(Y_hat_test, Y_prime_test)

def accuracy(pred_list, answ_list): #acc計算
    Alltag = 0
    anstag = 0

    for i in range(len(pred_list)):
        for j in range(len(pred_list[i])):
            if pred_list[i][j] == answ_list[i][j]:
                anstag += 1
            Alltag += 1
    print(f'accuracy: {anstag/Alltag}' )

#accuracy: 0.522244137628753


train_POS(os.path.join(dirname, "wiki-en-train.norm_pos")) #学習　trainデータを渡す
test_POS(os.path.join(dirname, "wiki-en-test.norm_pos")) #テスト　testデータを渡す
