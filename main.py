from preprocess import preprocess_text
from para import paraEntities
from Entities import  generateEntities
from f1_score import calculatescore
from matrix import confusionMatrix
from idf import idf

# Open  book in read mode
with open('Orwell-1949 1984.txt', 'r') as file:
    book = file.read()

# cleaning the text
modified_book=preprocess_text(book)

generateEntities(modified_book)


# trimming the three paragraphs from the book
para_1 = modified_book[0:1997]
para_2 = modified_book[4373:6098]
para_3 = modified_book[9823:11008]


# displaying entities of all the three paragraphs
para_1_entities=paraEntities(para_1)
para_2_entities=paraEntities(para_2)
para_3_entities=paraEntities(para_3)


# manually labelling the entities of all paras

para_1_manual_label=[("Eric Arthur Blair", "PERSON"), ("June ", "DATE"),("January","DATE"),("George Orwell", "PERSON")
                     ,("English", "NORP"),("Orwell","PERSON"),("Animal Farm","WORK_OF_ART"),
                     ("Nineteen EightyFour","WORK_OF_ART"),("The Road to Wigan Pier","WORK_OF_ART"),
                     ("England","GPE"),("Homage to Catalonia","WORK_OF_ART"),
                     ("Republican","NORP"),("Spanish Civil War","EVENT"),
                     ("George Orwell","PERSON"),("second","ORDINAL"),("British","NORP"),
                     ("Orwells","PERSON"),("English","LANGUAGE"),("Newspeak","LANGUAGE"),("day","DATE"),
                     ("April","DATE"),("thirteen","CARDINAL"),("Winston Smith","PERSON"),
                     ("one","ORDINAL"),("Winston","PERSON")]


para_2_manual_label=[("Winston", "PERSON"),("A kilometer","QUANTITY"),("the Ministry of Truth","ORG"),
                     ("London","GPE"),("Airstrip One","GPE"),("third","ORDINAL"),
                     ("Ocenia","GPE"),("London","GPE"),("chickenhouses", "LOC")]




para_3_manual_label=[("Winston", "PERSON"),("fourty years","DATE"),
                     ("a slummy quarter","LOC"),("Party","ORG"),("two dollars","MONEY")]





#calculating f1 score for all the three paras

print("Para1")
calculatescore(para_1_manual_label,para_1_entities)
print("Para2")
calculatescore(para_2_manual_label,para_2_entities)
print("Para3")
calculatescore(para_3_manual_label,para_3_entities)



# creating confusion matrixes for all the paras
confusionMatrix(para_1_manual_label,para_1_entities)
confusionMatrix(para_2_manual_label,para_2_entities)
confusionMatrix(para_3_manual_label,para_3_entities)

chapters = book.split('Chapter ')[1:]
idf(chapters)







