import regex as re
import numpy as np


'''
Simple-Bootstrapping
'''
class SimpleBootstrap:
    
    
    '''
    n_iter:     語の抽出を何回繰り返すかを決める
    rep_ptn:    語の抽出に用いる正規表現パターンの文字列
    words:      抽出された語の集合
    target_tdx: 3箇所ある抽出箇所のうちどこをターゲットとするか
    scores:     各語のスコア 繰り返しごとに保持する
    thr:        語を抽出する際のスコアの閾値
    idx:        ターゲットと手がかり表現のインデックス
    '''
    def __init__(self, n_iter=10, reg_ptn='', words0=[], words1=[], words2=[], target_idx=0, thr=0.3):
        self.n_iter = n_iter
        self.reg_ptn = reg_ptn
        self.scores = [['']*n_iter, ['']*n_iter, ['']*n_iter]
        self.words = [words0, words1, words2]
        self.target_idx = target_idx
        self.thr = thr
        self.idx = [[0,1,2],[1,2,0],[2,0,1]]
    
    
    '''
    extract関数で使用する
    手がかり表現の組を引数にとり，インデックスを返す辞書を作成する
    '''
    def make_idx(self, target_idx):
        idx1 = self.idx[target_idx][1]
        idx2 = self.idx[target_idx][2]
        return {(w1, w2):(len(self.words[idx2])*i+j) for i,w1 in enumerate(self.words[idx1]) for j,w2 in enumerate(self.words[idx2])}
    
    
    '''
    extract関数で使用する
    単語の共起頻度配列を正規化して単語の共起確率配列に変換する
    '''
    def to_prob(self, arr):
        return arr/np.repeat(np.sum(arr, axis=1).reshape(-1,1), arr.shape[1], axis=1)

    
    '''
    extract関数で使用する
    ターゲットのインデックスに応じて正規表現パターン文字列を適切に変換する
    '''
    def make_ptn(self, target_idx):
        result = re.search('(.*?)(\((?![?]).*?\))(.*?)(\((?![?]).*?\))(.*?)(\((?![?]).*?\))(.*)', self.reg_ptn)
        result = (result.group(1) + (result.group(2) if target_idx==0 else '({})'.format('|'.join(self.words[0])))
            + result.group(3) + (result.group(4) if target_idx==1 else '({})'.format('|'.join(self.words[1])))
            + result.group(5) + (result.group(6) if target_idx==2 else '({})'.format('|'.join(self.words[2])))
            + result.group(7))
        return result
    
    
    '''
    ターゲットと繰り返し時点とターゲット単語を指定してその時点でのスコアを返す
    ターゲット単語を指定しない場合はそのターゲットと繰り返し時点での全体のスコア辞書を返す
    '''
    def get_score(self, target_idx, iter_idx, word=None):
        if word is None:
            return self.scores[target_idx][iter_idx]
        else:
            return self.scores[target_idx][iter_idx][word]
    
    
    '''
    語の抽出を行う
    テキストを引数に受け取り，語の抽出，スコア計算，スコア閾値を超えた単語の格納を行う
    '''
    def extract(self, iter_idx, text, target_idxs):
        ptn = re.compile(self.make_ptn(target_idxs[0]))
        # 後で直す
        words_to_idx = self.make_idx(target_idxs[0])
        # ターゲット単語と手がかり表現の共起数
        freq_tgts = {}
        for sen in text.split('\n'):
            vals = ptn.findall(sen)
            # 引っかからなかったらスキップ
            if vals == []:
                continue
            if iter_idx==self.n_iter-1:
                print(sen)
            for val in vals:
                if val[target_idxs[0]] not in freq_tgts.keys():
                    # 初めての語が来たら辞書に追加
                    # エントロピーを用いるため微小値を足す
                    freq_tgts[val[target_idxs[0]]] = np.zeros(len(self.words[target_idxs[1]])*len(self.words[target_idxs[2]]))+(1e-7)
                freq_tgts[val[target_idxs[0]]][words_to_idx[(val[target_idxs[1]], val[target_idxs[2]])]] += 1
        keys = np.array(list(freq_tgts.keys()))
        # 正規化して確率値にする
        prob_tgts = self.to_prob(np.array(list(freq_tgts.values())))
        # スコア計算
        scores = {k:-v/(len(S)*len(E)) for k,v in zip(keys, np.diag(np.dot(prob_tgts, np.log2(prob_tgts).transpose())))}
        # スコア格納
        self.scores[target_idxs[0]][iter_idx] = scores
        # 語を格納
        self.words[target_idxs[0]] += [k for k,v in scores.items() if v>self.thr and k not in self.words[target_idxs[0]]]
        # ソート
        self.words[target_idxs[0]].sort(key=lambda x: -len(x))
        
    
    '''
    n_iter回単語の抽出を繰り返す
    '''
    def simple_bootstrap(self, text):
        for i in range(self.n_iter):
            for j in range(3):
                k = j+self.target_idx
                if k > 2:
                    k -= 3
                target_idxs = self.idx[k]
                self.extract(i, text, target_idxs)