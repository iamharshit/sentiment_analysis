import re
import nltk

s = '''Oh, U.S.A no,\" she\'s saying, \"our 400blender can\'t handle something this hard! harshitagarwal37@gmail.com https://stackoverflow.com/questions!!!!'''

pattern = r'''\d+|[A-Z][A-Z]+|http[s]?://[\w\./]+|[\w]+@[\w]+\.[\w]+|[\w\.]+|[\w]+|[-'a-z]+|[\S]+''' 

'''
Parses Numbers,Capital Letters, URLs, Email-ID, Abbreveations, Words, 's, doesn't losses information 
'''

l = re.findall(pattern, s) 	
print l	



