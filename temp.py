# coding: utf-8
import MeCab

#mt = MeCab.Tagger("mecabrc")
mt = MeCab.Tagger("-Ochasen")
res = mt.parseToNode("絵画は何枚ありますか")

while res:
    print res.surface
    #print res.feature
    res = res.next

