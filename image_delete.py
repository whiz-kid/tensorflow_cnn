import os

ext = ['jpg','jpeg','png']
path1 = ""
path1 = raw_input("Image Directory Name : ")
current_path = os.getcwd()
path1 = os.path.join(current_path,path1)
listing = os.listdir(path1)
for file in listing:
        if(os.path.isdir(os.path.join(path1,file))):
		path2 = os.path.join(path1,file)
		listing2 = os.listdir(path2)
		for image in listing2:
			# print(image)
			if(image.lower().endswith(tuple(ext)) == False):
				os.remove((os.path.join(path2, image)))

				
