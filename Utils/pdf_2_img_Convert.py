import sys
from pdf2image import convert_from_path
import os

def report(outputpath, file):
	with open(str(outputpath+"/"+"convertreport.txt"), "a") as report:
		report.write(str(file+" has been converted. \n"))
	print(file+" has been converted.")
	return

def convert2(outputpath, pages, file):
	pagecount = 1
	for page in pages:
		outputname  = str(file+'_'+str(pagecount)+'.png')
		page.save(str(outputpath+"/"+outputname), 'PNG')
		pagecount += 1
	return

def convert(path):
	'''Takes all files from a given directory with pdf files and turns them into jpg files. filename.pdf leads to filename_1.jpg, filename_2.png jpg.'''
	
	outputpath = path+'_output'
	if os.path.exists(outputpath):
		pass
	else:
		os.system("mkdir "+ outputpath)

	pages = convert_from_path(str(path), 500)
	dir_path, filename = os.path.split(path)
	convert2(outputpath, pages, filename)
	report(outputpath, filename)

	#print("All files are converted!")
	return outputpath

def main():
	if len(sys.argv) != 2:
		print("\"Usage of this function: convert.py input_path")
	if len(sys.argv) == 2:
		convert(sys.argv[1])	
	sys.exit(1)

if __name__ == '__main__':
	main()