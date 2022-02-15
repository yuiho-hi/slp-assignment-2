import os.path
from collections import deque

dirname = os.getcwd() #現在のファイルがあるフォルダのパスを取得
w = {}
w["shift"] = 0
w["left"] = 0
w["right"] = 0

def train_parse(train_file):
    save_list = []
    with open(train_file, mode='rt') as train_f:
        for line in train_f:
            line = line.replace('\n', '')
            if line:
                line = line.split('\t')
                save_list.append((line[0], line[1], line[3], line[6]))
            else:
                shift_reduce(deque(save_list), deque(save_list))
                save_list = []

def shift_reduce(queue, queue_for_corr, mode="train"):
    stack = [('0', "ROOT", "ROOT")]
    feats = 0
    unproc = unproc_word(queue)
    heads = {}

    stack_for_corr = [('0', "ROOT", "ROOT")]
    heads_corr = {}

    while len(queue) > 0 or len(stack) > 1:
        for feature, count in makefeats(stack, queue).items():
            if feature not in w:
                w[feature] = count
            feats += count
        score_s = w["shift"] * feats
        score_l = w["left"] * feats
        score_r = w["right"] * feats
        if score_s >= score_l and score_s >= score_r and len(queue) > 0:
            stack.append(queue.popleft())
            ans = "shift"

        elif score_r >= score_l:
            heads[int(stack[-1][0])] = stack[-2][0] #ID
            del stack[-1]
            ans = "right"
            
        else:
            heads[int(stack[-2][0])] = stack[-1][0] #ID
            del stack[-2]
            ans = "left"
            
        print(f'queue: {queue}')
        print(f'stack: {stack}')

        if len(stack_for_corr) >= 2 and stack_for_corr[-1][3] == stack_for_corr[-2][0] and unproc[stack_for_corr[-1][1] + " " + stack_for_corr[-1][0]] == 0:            
            unproc[stack_for_corr[-2][1] + " " + stack_for_corr[-2][0]] -= 1
            heads_corr[int(stack_for_corr[-1][0])] = stack_for_corr[-2][0]
            del stack_for_corr[-1]
            corr = "right"

        elif len(stack_for_corr) >= 3 and stack_for_corr[-2][3] == stack_for_corr[-1][0] and unproc[stack_for_corr[-2][1] + " " + stack_for_corr[-2][0]] == 0:            
            unproc[stack_for_corr[-1][1] + " " + stack_for_corr[-1][0]] -= 1
            heads_corr[int(stack_for_corr[-2][0])] = stack_for_corr[-1][0]
            del stack_for_corr[-2]
            corr = "left"

        else:
            stack_for_corr.append(queue_for_corr.popleft())
            corr = "shift"

        if  mode == "train" and ans != corr:
            w[ans] -= feats
            w[corr] += feats

    if  mode == "test":
        return sorted(heads.items()), sorted(heads_corr.items())


def makefeats(stack, queue):
    phi = {}
    if len(queue) >= 1:
        phi[stack[-1][2] + " " + queue[0][1]] = phi[stack[-1][2] + " " + queue[0][1]] + 1 if (stack[-1][2] + " " + queue[0][1]) in phi else 1 #P-1, W0
        phi[stack[-1][2] + " " + queue[0][2]] = phi[stack[-1][2] + " " + queue[0][2]] + 1 if (stack[-1][2] + " " + queue[0][2]) in phi else 1 #P-1, P0

        phi[stack[-1][1] + " " + queue[0][1]] = phi[stack[-1][1] + " " + queue[0][1]] + 1 if (stack[-1][1] + " " + queue[0][1]) in phi else 1 #W-1, W0
        phi[stack[-1][1] + " " + queue[0][2]] = phi[stack[-1][1] + " " + queue[0][2]] + 1 if (stack[-1][1] + " " + queue[0][2]) in phi else 1 #W-1, P0
    
    if len(stack) >= 2:
        phi[stack[-2][2] + " " + stack[-1][1]] = phi[stack[-2][2] + " " + stack[-1][1]] + 1 if (stack[-2][2] + " " + stack[-1][1]) in phi else 1 #P-2, W-1
        phi[stack[-2][2] + " " + stack[-1][2]] = phi[stack[-2][2] + " " + stack[-1][2]] + 1 if (stack[-2][2] + " " + stack[-1][2]) in phi else 1 #P-2, P-1

        phi[stack[-2][1] + " " + stack[-1][1]] = phi[stack[-2][1] + " " + stack[-1][1]] + 1 if (stack[-2][1] + " " + stack[-1][1]) in phi else 1 #W-2, W-1
        phi[stack[-2][1] + " " + stack[-1][2]] = phi[stack[-2][1] + " " + stack[-1][2]] + 1 if (stack[-2][1] + " " + stack[-1][2]) in phi else 1 #W-2, P-1
    
    return phi
        
def unproc_word(queue):
    unprocess = {}
    for i in range(len(queue)):
        unprocess[queue[i][1] + " " + queue[i][0]] = 0
        for j in range(len(queue)):
            if queue[i][0] == queue[j][3]: #ID == headの時
                unprocess[queue[i][1] + " " + queue[i][0]] += 1

    unprocess['ROOT 0'] = 1
    return unprocess
    
def test_parse(test_file):
    save_list_test = []
    heads_pred = []
    heads_answ = []
    with open(test_file, mode='rt') as test_f:
        for line in test_f:
            line = line.replace('\n', '')
            if line:
                line = line.split('\t')
                save_list_test.append((line[0], line[1], line[3], line[6]))
            else:
                heads, heads_corr = shift_reduce(deque(save_list_test), deque(save_list_test), mode="test")
                save_list_test = []
                heads_pred.append(heads)
                heads_answ.append(heads_corr)

    accuracy(heads_pred, heads_answ)

def accuracy(heads, heads_answ):
    all_head = 0
    ans_head = 0

    for i in range(len(heads)):
        if heads[i][1] == heads_answ[i][1]:
                ans_head += 1
        all_head += 1
    print(f'accuracy: {ans_head/all_head}')

#accuracy: 0.07

train_parse(os.path.join(dirname, "mstparser-en-train.dep")) #学習　trainデータを渡す
test_parse(os.path.join(dirname, "mstparser-en-test.dep")) #テスト　testデータを渡す