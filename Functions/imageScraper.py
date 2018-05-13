import urllib.request
import sys
import random
import scipy.ndimage

def getURLs(URL): #returns a list of URLs for a specific imagnet URL
	URLs = []
	download_data = urllib.request.urlopen(URL)
	for line in download_data:
		cur_URL = line.decode("utf-8")
		cur_URL = cur_URL.strip('\n')
		URLs += [cur_URL]
	return URLs


# file_type = "." + URLs[0].split('.')[-1]
def testOpenImage(path):
	try:
		image = scipy.ndimage.imread(path,flatten=False)
	except OSError:
		print("An OS Error occured when trying to open image",path)
		return 404
	return 

def downloadImage(URL,filename,filepath):
	try:
		image = urllib.request.urlopen(URL,timeout=5).read()
		image_size = sys.getsizeof(image)
		assert (image_size > 2084), "Image Size Invalid" #2084 is the size of image not found image. Anything less than 2084 is too small and will need to be stretched too much 
		file = open(filepath + filename + ".jpg", "wb")
		file.write(image)
		file.close()
		assert(testOpenImage(filepath + filename + ".jpg") != 404), "File Corrupted"
	except (urllib.error.HTTPError, urllib.error.URLError, AssertionError):
		print("Error in downloadImage", filepath + filename)
		print(URL)
		return 404

	except Exception as e:
		print("A foreign error occured:",e)
		print("At this URL:",URL)
		return 404
	return None

def downloadImages(num_of_images,object_name,filepath, URLs):
	assert (num_of_images < len(URLs)), "Not Enough URLs to download num_of_images"
	i = 0
	num_of_images -= 1 #counts exclusively now
	while True:
		image_number = abs(i - num_of_images)
		cur_filename = object_name + str(image_number) 
		cur_URL = URLs[i]
		if downloadImage(cur_URL,cur_filename,filepath) == 404:
			num_of_images += 1
			assert (num_of_images < len(URLs)), "Not Enough URLs to download num_of_images"
		if i == num_of_images: #counts exclusively
			break
		i += 1


def downloadRandomImages(num_of_images,object_name,filepath):
	random_image_file = open("fall11_urls.txt", encoding="ISO-8859-1")
	random_image_file = random_image_file.readlines()
	random.shuffle(random_image_file)
	i = 0
	num_of_images -= 1 #counts exclusively now
	while True:
		cur_line = random_image_file[i]
		cur_line = cur_line.strip("\n")
		cur_line = cur_line.split("http://")[-1] 
		cur_URL = "http://" + cur_line
		image_number = abs(i - num_of_images)
		cur_filename = "not" + object_name + str(image_number)
		if downloadImage(cur_URL,cur_filename,filepath) == 404:
			num_of_images += 1
		if i == num_of_images: #counts exclusively
			break
		i += 1

def main():
	positive_image_filepath = "Logistic_Regression_Data/cows/"
	negative_image_filepath = "Logistic_Regression_Data/notcows/"
	cow_images_URL  = "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01887787"
	cow_images_URLs = getURLs(cow_images_URL)
	object_name = "cow"
	num_of_images = 500

	downloadImages(num_of_images,object_name,positive_image_filepath,cow_images_URLs)
	downloadRandomImages(num_of_images,object_name,negative_image_filepath)


print("Imaged Scraper Started :O")
main()
print("Image Scraper Finisehd ;)")