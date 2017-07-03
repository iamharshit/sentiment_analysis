from collections import Counter
import glob
import re


files = glob.glob('/home/anup/SplitAudioFiles/train-clean-20-wav/*.txt')
#files = glob.glob('/home/anup/SplitAudioFiles/office_speech_banking_data_without_test/*.txt')

remove_words = ['of', 'a', 'an', 'is', 'was', 'ok', 'he', 'she', 'i', 'am', 'so', 'be', 'it', 'for', 'if', 'the', 'by', 'has', 'have', 'to', 'can', 'this', 'that', 'in','on', 'you', 'not', 'me','we','are', 'my', 'at', 'will' , 'as', 'or']


output_single = open('frequency_single_voicelogs.txt','w')
output_bigram = open('frequency_bigram_office_speech.txt','w')
output_trigram = open('frequency_trigram_office_speech.txt','w')
def create_list():
	allWords= []
	for file in files:
		#words = re.findall(r'\w+', open(file).read())
		trans = open(file).read().lower().split()			
		#print len(words) ,len (trans)
		allWords.extend(trans)
	#Words = list(set(allWords) - set(remove_words))
	Words = filter(lambda x: x not in remove_words, allWords)
	#print len(Words)
	allWords = Words
	return allWords

def getUniqueWords(words):
    ll = Counter(words)	
    print (ll.most_common(10))
    return ll.most_common()

def bi_gram():
	total_bigrams = []
	for file in files:
		trans = open(file).read().split()			
		bigrams = [b for b in zip(trans[:-1], trans[1:])]
		total_bigrams.extend(bigrams)
	unique = getUniqueWords(total_bigrams)
	for k, v in unique:
		output_bigram.write(str(v) +'	' + str(k))
		output_bigram.write('\n')
	print ('total bigrams: ' + str(len(total_bigrams)) , 'unque bigram: ' +str(len(unique)))

def tri_gram():
	total_trigrams = []
	for file in files:
		trans = open(file).read().split()			
		trigrams = [b for b in zip(trans[:-2], trans[1:-1], trans[2:])]
		total_trigrams.extend(trigrams)
	unique = getUniqueWords(total_trigrams)
	for k, v in unique:
		output_trigram.write(str(v) +'	' + str(k))
		output_trigram.write('\n')
	print( 'total trigrams: ' +str(len(total_trigrams)) , 'unque trigram: ' +str(len(unique)))
	

def main():
	allWords = create_list()
	unique = getUniqueWords(allWords)
	for k, v in unique:
		output_single.write(str(v) +', ' + str(k))
		output_single.write('\n')
	print ('total words: ' + str(len(allWords)),  'unique word :' + str(len(unique)))
	#bi_gram()
	#tri_gram()

if __name__ == "__main__":
    main()

