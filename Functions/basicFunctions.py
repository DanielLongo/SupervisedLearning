import urllib.request
import numpy as np

def getURLs(URL):
	URLs = []
	download_data = urllib.request.urlopen(URL)
	for line in download_data:
		try:
			cur_URL = line.decode("utf-8")
			print(cur_URL)
			cur_URL = cur_URL.strip('\n')
			URLs += [cur_URL]
		except ValueError:
			print("ERROR")
			continue
	return URLs


# file_type = "." + URLs[0].split('.')[-1]

def downloadImages(URLs,counter=0,fileName="cow"):
	file_path = 'Logistic_Regression_DataB/'
	for URL in URLs:
		try:
			file_name = fileName+str(counter)
			f = open(file_path+file_name,'wb')
			file = urllib.request.urlopen(URL,timeout=5).read()
			# print(file)
			if file == None or file == '':
				print("HEREHREHREHR")
			f.write(file)
			f.close()
			counter += 1
			print(str(counter/len(URLs))+"%")
			print(counter)
		except:
			print("An error occured")
			continue
	return 

URL = "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01887787"
# downloadImages(getURLs(URL)[0:10])

def getRandomImages(num_of_images):
	file = open("fall11_urls.txt")
	counter = 0
	for line in file:
		if counter > num_of_images:
			break
		line = line.split("http://")[-1]
		cur_URL = "http://"+ line
		if downloadImages([cur_URL],counter=counter,fileName="notCow") == 404:
			print("got the error")
			num_of_images += 1
		counter += 1

#getRandomImages(50)

def crossEntropyLoss(a,Y):
    num_of_examples = len(a)
    loss = (-1/num_of_examples)*np.sum(np.dot((1-Y),np.log(1-a)) + np.dot((Y, np.log(a))))
    return loss


# print(URLs[0])
# print(urllib.request.urlretrieve(URLs[0]),"testCowPic")

print("Finisehd ;)")

