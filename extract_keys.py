keyValues = {"CompanyName":"","ReceiptNumber":"","Tax":"","Amount":"","ReceiptDate":"","CurrencyType":""}

import json
from math import degrees
from re import search,IGNORECASE, sub
from json import loads
import numpy as np
import cv2
from nltk import edit_distance
from string import punctuation
from math import degrees, atan
import os
import datetime
import time
import base64
import requests

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="teamexpense-316820-f4964d3162d2.json"

def Tr2EngLower(data):
    data = sub("[İíìîÍÌÎıİi̇I]","i",data)
    data = sub("[ÜüúÚ]","u",data)
    data = sub("[Şş]","s",data)
    data = sub("[Öö]","o",data)
    data = sub("[Ğğ]","g",data)
    data = sub("[ÇçčČ]","c",data)
    return data.lower()


def fixString(data):
    data = sub("[Ú]","U",data)
    data = sub("[Č]","C",data)
    return data

def removePunctiations(text):
    excPuncList = [".","#",","]

    for punc in punctuation:
        if punc not in excPuncList:
            text = text.replace(punc,"")
    
    return text

def detect_text(img):
    global counter

    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=img)
    response = client.text_detection(
    image=image,
    image_context={"language_hints": ["tr"]},  # Turkish
    )
    texts = response.text_annotations
    response_json = vision.AnnotateImageResponse.to_json(response)
    response = loads(response_json)
    return response, texts


def findCompanyName(text,keyValues):
    companyNameKeys = {"CompanyName": ["ltd","sti","gid","turz","tic","paz","a s","san","dag","sdn","bhd"]}
    companyNameControlKeys = {"control":["tel","fax","adres","no","sk","sok","mh","cad","cd","bul","adress"]}
    
    line_by_line = text.split("\n")

    line_lim = 6
    companyName = None

    for i,line in enumerate(line_by_line[:line_lim]):   
        #Stringi türkçe büyükten ingilizce küçüğe çevirip
        #Keylere göre bir arama yapılıyor 
        #Ve bir önceki satırla keylerin bulunduğu satır alınıyor
        line = Tr2EngLower(line)
        line = line.replace("."," ")
        line = line.replace(":"," ")
        for key in companyNameKeys["CompanyName"]:
            match = search("\s"+key+"\s",line)
            if(match and companyName is None):
                
                #şirket isminde adresin ve iletişim bilgilerinin geçmemesi gerekli 
                for controlKey in companyNameControlKeys["control"]:
                    firstLine= Tr2EngLower(line_by_line[i-1])
                    firstLine = firstLine.replace("."," ").replace(":"," ")
                    match = search(controlKey,firstLine)
                    if(match):
                        break

                if(match):
                    companyName = line_by_line[i]

                else:
                    companyName = line_by_line[i-1] + " " + line_by_line[i]
                    
                
                companyName = ''.join([i for i in companyName if not i.isdigit()])
                companyName = companyName.strip()
                keyValues["CompanyName"] = fixString(companyName)

        #Eğer hiçbir match bulunamazsa ilk satırı şirket ismi kabul ediyor 
    
    secondLineControl = False
    firstLineControl = False

    if(companyName is None):

        for controlKey in companyNameControlKeys["control"]:
            firstLine = Tr2EngLower(line_by_line[0])
            firstLine = firstLine.replace("."," ").replace(":"," ")

            secondLine = Tr2EngLower(line_by_line[1])
            secondLine = secondLine.replace("."," ").replace(":"," ")

            firstLineMatch = search(controlKey,firstLine)
            secondLineMatch = search(controlKey,secondLine)

            if(firstLineMatch):
                secondLineControl = True
            if(secondLineMatch):
                firstLineControl = True

        if secondLineControl:
            companyName = line_by_line[1]
            
        if firstLineControl:
            companyName = line_by_line[0]
        
        else:
            companyName = line_by_line[0] + " " + line_by_line[1]
        
        companyName = ''.join([i for i in companyName if not i.isdigit()])
        companyName.strip()
        companyName = companyName.replace("  "," ")
        keyValues["CompanyName"] = fixString(companyName)
    

def findDate(newText,keyValues):
    # Oluşturulan yeni Text içerisinden Tarih ve Saat bilgileri çekiliyor.
    tarih = ["/","-","\\","\."]
    text = newText.replace("\n"," ")
    saat = "00:00"
    date = "00/00/0000"
    match = None

    for i in tarih:
        match = search(fr"\b\d\d{i}\d\d{i}\d\d\d\d\b",text)
        if match:
            date = match.group()
            break

    if not match:
        tarih = ["/","-","\\","\."]

        for i in tarih:
            match = search(fr"\b\d\d{i}\d\d{i}\d\d\b",text)
            if match:
                date = match.group()
                break

    match = search(r"\b\d{2}:\d{2}:\d{2}\b",text)
    if match:
        saat = match.group()
    else:
        match = search(r"\b\d{2}:\d{2}\b",text)
        if match:
            saat = match.group()
    
    for char in tarih:
        date = date.replace(char," ")
    date = date.replace("."," ")

    try:
        # Saat 00:00 or 00:00:00 kontrolü
        if len(saat.split(":")) == 2 :
            datetimeObject = datetime.datetime.strptime(date+' '+saat,'%d %m %Y %H:%M')
        
        if len(saat.split(":")) == 3 :
            datetimeObject = datetime.datetime.strptime(date+' '+saat,'%d %m %Y %H:%M:%S')
        
        formatted_datetime = datetimeObject.isoformat()
        json_datetime = json.dumps(formatted_datetime)
        keyValues["ReceiptDate"] = json_datetime
    
    except Exception as e:
        print(e)
        # fişlerdeki tarih 00-00-0000 yerine 00-00-00 olursa except durumuna giriyor
        try : 
            date_list = date.split(" ")
            tempDate = ""
            date_list[-1] = '20' + date_list[-1]
            for d in date_list : 
                tempDate += d + " " 
            
            tempDate = tempDate.strip()
            # Saat 00:00 or 00:00:00 kontrolü

            if len(saat.split(":")) == 2 :
                datetimeObject = datetime.datetime.strptime(tempDate+' '+saat,'%d %m %Y %H:%M')
            
            if len(saat.split(":")) == 3 :
                datetimeObject = datetime.datetime.strptime(tempDate+' '+saat,'%d %m %Y %H:%M:%S')
            
            
            formatted_datetime = datetimeObject.isoformat()
            json_datetime = json.dumps(formatted_datetime)
            keyValues["ReceiptDate"] = json_datetime

        except :
            keyValues["ReceiptDate"] = None



def findKeyValuesWordSimilarity(newText,TurkishKeys,keyValues):
    for line in newText.split("\n"):
        for dic, key in TurkishKeys.items():
            for key in TurkishKeys[dic]:
                line = line.replace(":"," ")
                line = sub("[,]",".",line)
                line = removePunctiations(line)
                words = line.split(" ")
                words = [i for i in words if i]
                nextKey = False

                if " " in key and len(key.split(" ")) <= len(words):
                    for i in range((len(words) - len(key.split(" "))) - 1):
                        word = " ".join([words[i + a] for a in range(0,len(key.split(" ")))])
                        wordSimilarityScore = edit_distance(word,key)

                        if wordSimilarityScore <= 1:
                            keyValues[dic] = " ".join([value for value in words[i+1:] if not value.replace(".","").replace("#","").isalpha() and not "%" in value])
                            nextKey = True
                            if keyValues[dic] != "":
                                TurkishKeys[dic] = []
                            break
                
                else:
                    for i in range(len(words)):
                        word = words[i]
                        wordSimilarityScore = edit_distance(word,key)
                        
                        if wordSimilarityScore <= 1:
                            keyValues[dic] = "".join([value for value in words[i+1:] if not value.replace(".","").replace("#","").isalpha() and not "%" in value])
                            nextKey = True
                            if keyValues[dic] != "":
                                TurkishKeys[dic] = []
                            break
                
                if nextKey:
                    break


def findKeyValuesSearch(newText,TurkishKeys,keyValues):
    for line in newText.split("\n"):
        for dic in TurkishKeys:
            for key in TurkishKeys[dic]:
                line = line.replace(":"," ")
                line = removePunctiations(line)
                match = search(rf"\b{key}\b", line, IGNORECASE)
                if match:
                    line = line[match.end():]
                    line = line.replace(":"," ")
                    line = sub("[,]",".",line)
                    words = line.split(" ")
                    words = [i for i in words if i]
                    
                    if len(words) == 1:
                        if not words[0].isalpha() and all(words[0]) != punctuation:
                            keyValues[dic] = words[0]
                        
                    elif len(words) > 1:
                        keyValues[dic] = "".join([value for value in words if not value.replace(".","").replace("#","").isalpha() and not "%" in value])
                    
                    if keyValues[dic] != "":
                        TurkishKeys[dic] = []


def findCurrency(newText,keyValues):
    if newText.find("$") > -1:
        keyValues["CurrencyType"] = "USD"
    elif newText.find("€") > -1:
        keyValues["CurrencyType"] = "EUR"
    else:
        keyValues["CurrencyType"] = "TL"


def extractKeys(newText):
    TurkishKeys = {"ReceiptNumber":["fis no","fisno","fatura no"],"Tax":["topkdv","top.kdv","toplam kdv","kdv"],"Amount":["toplam","top"]}
    EnglishKeys = {"ReceiptNumber":["inv no","invoice no","doc no","slip no","check #","bill #","inv#","cb#"],"Tax":["total included gst","total gst","gst payable","tax rm"],"Amount":["total sales inclusive of gst","total inclusive of gst","total payable","total sales","total rm","total amount","total rounded","due"]}
    
    
    findCompanyName(newText,keyValues)
    findDate(newText,keyValues)
    findCurrency(newText,keyValues)

    # Türkçe karakterler ingilizce karakterlere dönüştürülüyor.
    newText = Tr2EngLower(newText)

    # Oluşturulan satır satır yazdırılan text arasından önceden belirlenen key wordler aranıyor
    # Daha sonra bulunan kategorilerdeki değerler regular expression yöntemiyle bulunarak alınıyor
    # Arama işleminde kategorilerden biri bulunmazsa kelime benzerliği yöntemi uygulanıyor

    text = newText.replace("\n"," ")
    text = removePunctiations(text)
    TurkishWordSimilarityKeys = TurkishKeys.copy()
    EnglishWordSimilarityKeys = EnglishKeys.copy()
    searchingKeys = {}
    englishKeyNum = 0

    for dic in TurkishKeys:
        turkishKeyFound = False
        for key in TurkishKeys[dic]:
            match = search(rf"\b{key}\b", text, IGNORECASE)

            if match:
                searchingKeys[dic] = [key]
                del TurkishWordSimilarityKeys[dic]
                turkishKeyFound = True
                break
        
        if not turkishKeyFound:
            for key in EnglishKeys[dic]:
                match = search(rf"\b{key}\b", text, IGNORECASE)

                if match:
                    englishKeyNum += 1
                    searchingKeys[dic] = [key]
                    del EnglishWordSimilarityKeys[dic]
                    break
        
    
    if englishKeyNum == 0:
        if len(TurkishWordSimilarityKeys) > 0:
            findKeyValuesWordSimilarity(newText,TurkishWordSimilarityKeys,keyValues)
        
    else:
        if len(EnglishWordSimilarityKeys) > 0:
            findKeyValuesWordSimilarity(newText,EnglishWordSimilarityKeys,keyValues)

    if len(searchingKeys) > 0:
        findKeyValuesSearch(newText,searchingKeys,keyValues)


    if(keyValues["Amount"]=="" or keyValues["Amount"]== " "):
        findKeyValuesSearch(newText,{"Amount":["toplam tutar","tutar","nakit"]},keyValues)
    

    if keyValues["Amount"]:
        keyValues["Amount"] = "".join([i for i in keyValues["Amount"] if not i.isalpha() and i != "€"])

    if keyValues["Tax"]:
        keyValues["Tax"] = "".join([i for i in keyValues["Tax"] if not i.isalpha() and i != "€"])

    return keyValues
