AWS file structure

```
/nlp_data
|
└── NLP_dhruv_harshit
    |
    ├── DataStats_ofChat
    |   |
    │   ├── words_not_in_glove_6B.txt
    │   ├── words_not_in_glove840_cased.txt
    │   ├── words_not_in_glove840.txt
    │   ├── words_not_in_glove_twitter_cased.txt
    │   └── words_not_in_glove_twitter.txt
    |
    ├── emotion_analysis
    |   |    
    │   ├── emotion_analysis.ipynb
    │   └── imdb_emotions.csv
    |      
    └── sentiment_analysis
        |   
        ├── Dataset (Contains the chat dataset & glove)
        |   |
        │   ├── beam_cable_google_tagged_data.json
        │   ├── droom_google_tagged_data.json
        |   |
        │   ├── data.csv
        │   ├── train.csv
        │   ├── test.csv
        │   ├── valid.csv        
        |   |
        │   ├── glove.6B.300d.txt
        │   ├── glove.840B.300d.txt 
        |   |
        │   ├── test_x.pkl
        │   ├── test_y.pkl
        │   ├── train_x.pkl
        │   ├── train_y.pkl
        │   ├── val_x.pkl
        │   └── val_y.pkl
        |   |
        |   ├── Message_VS_sentiment&Emotion - Sheet1.csv
        |   
        ├── GloveDataDistribution_4B_and_840B
        │   |
        │   ├── glove6B
        │   │   ├── conv_length2count.jpg
        │   │   ├── create_embedding.ipynb
        │   │   ├── glove_vocab_comparison.ipynb
        │   │   ├── no.ofMsgs_vs_scores_afterSubsampling.png
        │   │   ├── no.ofMsgs_vs_scores_beforeSubsampling.png
        │   │   ├── no.ofMsgs_vs_scores_both.png
        │   │   ├── plot_NoOfMsgAboveThreshold_vs_NoOfConversations_HA.ipynb
        │   │   ├── plot_NoOfMsg_vs_NoOfConversations_And_Polarity_vs_NoOfConversations_DS.ipynb
        │   │   ├── samarth_data.ipynb
        │   │   ├── samarth_data.py
        │   │   ├── score2freq.png
        │   │   
        │   ├── glove6B_DataInput_and_DataStats_and_DataDistributionPlot_DS.ipynb
        │   ├── glove6B_DataStats_HA.ipynb
        │   ├── glove6B_OldPreprocessor.ipynb
        │   ├── glove840_vocab_comparison.py  
        │   ├── vocab_of_chat.txt
        │   └── words_of_glovetwitter6B.txt
        |
        |   
        ├── milestone-1 (IMDB & Feed forward Net)
        │   ├── data
        │   │   ├── labeledTrainData.tsv
        │   │   └── testData.tsv
        │   ├── M1_code .ipynb
        │   └── sentiment_nn.py
        |
        ├── milestone-2 (IMDB & RNN)
        |   |
        │   ├── M2_code-harshit.ipynb
        │   ├── M2_code.ipynb
        │   ├── M2_code.py
        |
        ├── milestone-3 (Chat & RNN & GloVe.6B)
        |   |
        │   ├── data_input.ipynb
        │   ├── GeneralityCheck_HS.ipynb
        │   ├── glove840_vocab_comparison.py
        │   ├── M2_code-harshit-without-dropout.ipynb
        │   ├── M2_code-harshit-without-dropout-new+(5).ipynb
        │   ├── M2_code-harshit-without-dropout-new.ipynb
        │   ├── M2_code.ipynb
        │   └── unique_words.py
        |
        ├── milestone-3.5 (Chat & LR)
        │   ├── LR.ipynb
        │   └── LR.py
        |   
        ├── milestone-4 (Chat & GLoVe.840B)
        |   | 
        │   ├── LeftPadding_6B.ipynb
        │   ├── main_hyperparameter_tuning.ipynb
        │   ├── main_hyperparameter_tuning.py
        │   ├── make_pkl_files.py
        │   ├── Make_TrainValTest_pickle.ipynb
        │   ├── samarth_data.ipynb   
        │   │   
        │   └── wrong_result.ipynb
        │   
        ├── Results   (Contains results of Hyperparameter tuning)
        │   ├── Result6B    
        │   ├── Result840B            
        |   ├── Results.md
        |   ├── FinalReport = https://docs.google.com/document/d/1HlEqMnx75JYu33f9TFHU_EpVRqIOqaX05_EnjXmBVUk/edit
        |   ├──result sheet = https://docs.google.com/spreadsheets/d/1CdMQFgi3VF_tO60C0EBcSVn9nyZDFxIPCRVJ5xw9G2Y/edit 
        |
        ├── FinalCode-part0.ipynb
        ├── FinalCode-part1.ipynb
        ├── FinalCode-part2.ipynb   
```
