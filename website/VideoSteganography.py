#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import cv2 
from matplotlib import pyplot as plt
import random
import math
import time
import os
import subprocess
import shutil
import glob
import time
import moviepy.editor as mp 
import wave
import errno,stat


# In[7]:



def handleRemoveReadonly(func, path, exc):
  excvalue = exc[1]
  if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
      os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
      func(path)
  else:
      raise

def encryptCustom(videoFile,textFile):
    word_size = 16 # Size of the number in byte format
    secret_key = 567 # Secret key already known by the program 
    byte_len=8
    encryptVideo = ""
    original_text = open(textFile,'r').read()
    
    video=cv2.VideoCapture(videoFile)
    num_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    text_list=original_text.split()
    final_text_list=[]
    for t in text_list: 
        if len(t)>10:
            final_text_list.append(t[:8]+"-")
            final_text_list.append(t[8:])
        else:
            final_text_list.append(t)
    text=" ".join([x for x in final_text_list])
    text_length=len(text.split())
    random_frame_array=[]
    for i in range(text_length):
        random_frame_array.append(int(random.uniform(100,num_of_frames-100)))
    random_frame_array.sort()
    random_frame_array_copy=random_frame_array.copy()
    random_frame_array_copy.insert(0,len(random_frame_array))

    # Load video files and change it into the wav format
    video = mp.VideoFileClip(videoFile) 
    audio = video.audio
    audio.write_audiofile("Audio.wav")
    audio = wave.open("Audio.wav", mode='rb')
    
    # Load bytes from the audio file
    frame_data = audio.readframes(audio.getnframes())
    audio.close()
    os.remove("Audio.wav")
    frame_bytes = bytearray(list(frame_data))
    
    # Frame data to be written first element is number of frame object
    frame_data = random_frame_array_copy
    number_element = frame_data[0]
    
    # Convert the frame into bit data of word size and then store it in the audio file
    frame_data = ''.join(list(map(lambda number: '{0:0{width}b}'.format(number,width=word_size),frame_data)))
    frame_data = list(map(int,frame_data))
    frame_bytes_length = len(frame_bytes)
    
    number_random_position = np.random.RandomState(seed=secret_key).choice(frame_bytes_length,word_size) # Get word_size random positions
    
    # Peform steganography encrpt the number of frames in random places
    for i, bit in enumerate(frame_data[:word_size]):
        frame_bytes[number_random_position[i]] = (frame_bytes[number_random_position[i]] & 254) | bit
    
    # Remove encoded values from frame data and get random location to write data
    frame_data = frame_data[word_size:]
    random_bit_location = np.random.RandomState(seed=secret_key*number_element).choice(frame_bytes_length,word_size*(number_element+1))
    random_bit_location = list(set(random_bit_location) - set(number_random_position))[:word_size*number_element]

    # Peform steganography and save the data in the encrypted format
    for i, bit in enumerate(frame_data):
        frame_bytes[random_bit_location[i]] = (frame_bytes[random_bit_location[i]] & 254) | bit
    
    # Generate modified frames
    modified_frames = bytes(frame_bytes)
    
    # Store the final audio file
    final_audio = wave.open("Encrypted.wav","wb")
    final_audio.setparams(audio.getparams())
    final_audio.writeframes(modified_frames)
    final_audio.close()
    
    video=cv2.VideoCapture(videoFile)
    num_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


    #EXTRACT AND ENCODE FRAMES
    frame_copy_list=[]
    position_properties_list=[]
    prev_frame_copy_list=[]
    text_list=text.split()
    t_dict={}
    #print(text_list)
    frame_dict={}
    prev_frame_dict={}
    for index,text_element in enumerate(text_list):
        #frame=cv2.imread("frame_{}.jpg".format(i+1))
        #video.set(1, random_frame_array[index])
        video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_array[index])
        success_flag, frame = video.read()
        frame_copy=frame.copy()
        frame_data=' '.join([format(ord(i),"08b") for i in text_element]) #convert to 8-bit binary
        #frame_data = list(map(int,frame_data))
        frame_data=frame_data.split()
        #print(frame_data)

        corner=int(random.uniform(1,5))
        word_len=len(frame_data)
        columns_needed=math.ceil(word_len*(byte_len+1)/3)
        if corner==1:
            row=0
            col=0
        if corner==2:
            row=0
            col=frame.shape[1]-(columns_needed+1)
        if corner==3:
            row=frame.shape[0]-1
            col=0
        if corner==4:
            row=frame.shape[0]-1
            col=frame.shape[1]-(columns_needed+1)
        #print(row,columns_needed,word_len,byte_len,frame[row,col])


#         print(corner,word_len)

        r=row
        if col!=0:
            c=col
        else:
            c=-1

        for i,element in enumerate(frame_data):
            for j in range(byte_len):
                if j%3==0:
                    c+=1
                    #print(c)
                #print(frame_data[i][j],frame[r,c,j%3])
                curr_pixel=frame[r,c,j%3]
                new=curr_pixel
                if frame_data[i][j]=='0' and curr_pixel%2!=0:
                    frame_copy[r,c,j%3]-=1
                elif frame_data[i][j]=='1' and curr_pixel%2==0:
                    if curr_pixel!=0:
                        frame_copy[r,c,j%3]-=1
                    else:
                        frame_copy[r,c,j%3]+=1
                #print(frame_data[i][j],r,c,frame_copy[r,c,j%3]%2)
            #print()
        frame_copy_list.append(frame_copy)
        position_properties_list.append((corner,word_len))

        t_dict[random_frame_array[index]]=frame_copy
        frame_dict[random_frame_array[index]]=frame_copy

        #write row number in the frame
        corner_t=format(corner,"08b")

        #print(row_num,row,len(row_num),byte_len,random_frame_array[index]-1)

        #video.set(1,(random_frame_array[index]-1))
        video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_array[index]-1)
        success_flag, frame = video.read()

        prev_frame_copy=frame.copy()                 
        #print(prev_frame_copy[0,0],prev_frame_copy[0,1])

        r=0
        c=-1

        for j in range(byte_len):
            if j%3==0:
                c+=1
                #print(c)
            #print(frame_data[i][j],frame[r,c,j%3])
            curr_pixel=frame[r,c,j%3]
            new=curr_pixel
            if corner_t[j]=='0' and curr_pixel%2!=0:
                prev_frame_copy[r,c,j%3]-=1
            elif corner_t[j]=='1' and curr_pixel%2==0:
                if curr_pixel!=0:
                    prev_frame_copy[r,c,j%3]-=1
                else:
                    prev_frame_copy[r,c,j%3]+=1
            #print(corner_t[j],r,c,curr_pixel,prev_frame_copy[r,c,j%3]%2)

        w_len=format(word_len,"08b")

        r=0
        c=2

        for j in range(byte_len):
            if j%3==0:
                c+=1
                #print(c)
            #rint(frame_data[i][j],frame[r,c,j%3])
            curr_pixel=frame[r,c,j%3]
            new=curr_pixel
            if w_len[j]=='0' and curr_pixel%2!=0:
                prev_frame_copy[r,c,j%3]-=1
            elif w_len[j]=='1' and curr_pixel%2==0:
                if curr_pixel!=0:
                    prev_frame_copy[r,c,j%3]-=1
                else:
                    prev_frame_copy[r,c,j%3]+=1
            #print(w_len[j],r,c,curr_pixel,prev_frame_copy[r,c,j%3]%2)   

        prev_frame_copy_list.append(prev_frame_copy)
        t_dict[random_frame_array[index]-1]=prev_frame_copy
        prev_frame_dict[random_frame_array[index]-1]=prev_frame_copy
        #print("-----------")
    height, width, layers = frame.shape
    frameSize = (width,height)
    frameSize
    video.release() 
    cv2.destroyAllWindows()
    
    #EMBED ENCODED FRAMES INTO A VIDEO
    start=time.time()
    video=cv2.VideoCapture(videoFile)
    fps=video.get(cv2.CAP_PROP_FPS)
    num_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if not os.path.exists('static/frames_encode'):
        os.makedirs('static/frames_encode')

    os.chdir('static/frames_encode')

    for i in range(num_of_frames):
        success_flag, frame = video.read()
        if i not in list(frame_dict) and i not in list(prev_frame_dict):
            pass
        else:
            #print(i)
            if i in list(frame_dict):
                frame=frame_dict[i]
            else:
                #print(prev_frame_dict[i][0][0],prev_frame_dict[i][0][1],prev_frame_dict[i][0][2])
                #print(prev_frame_dict[i][0][3],prev_frame_dict[i][0][4],prev_frame_dict[i][0][5])
                frame=prev_frame_dict[i]

        cv2.imwrite("frame{}.png".format(i),frame)

    video.release() 
    cv2.destroyAllWindows()

    #subprocess.getoutput(['ffmpeg','-framerate','{}'.format(fps),'-i','frames_encode/frame%d.png','-codec','copy','output.mp4'])
    os.chdir('../..')
    subprocess.run('ffmpeg -y -framerate {} -i static/frames_encode/frame%d.png -vcodec ffv1 -level 3 output.mkv'.format(fps))

    subprocess.getoutput(['ffmpeg','-y','-i','output.mkv','-i','Encrypted.wav','-c:v','copy','-c:a','copy','static/final_output.mkv'])

    os.remove('output.mkv')
    os.remove("Encrypted.wav")

    shutil.rmtree('static/frames_encode',ignore_errors=False, onerror=handleRemoveReadonly) 
#     os.rmdir("static/frames_encode")
#     encryptVideo = "final_output.mkv"
    
#     return encryptVideo

#DECRYPTION

def decryptCustom(videoFile):
    word_size = 16 # Size of the number in byte format
    secret_key = 567 # Secret key already known by the program 
    byte_len=8
    # Load video files and change it into the wav format
    subprocess.run(['ffmpeg','-y','-i',videoFile,'-c','copy','Audio.wav'])

    # Load the final audio file and extract frames from it
    audio = wave.open("Audio.wav", mode="rb")
    frame_data = audio.readframes(audio.getnframes())
    frame_bytes = bytearray(list(frame_data))
    audio.close()
    os.remove("Audio.wav")
    #print(len(frame_bytes),type(frame_bytes[0]))

    frame_bytes_length = len(frame_bytes) # Get number of frames
    number_random_position = np.random.RandomState(seed=secret_key).choice(frame_bytes_length,word_size) # Get word_size random positions
    #print(number_random_position)

    # Extract bits from the message and get number of frames from data
    extracted = [frame_bytes[number_random_position[i]] & 1 for i in range(word_size)]
    number_element = int(str("".join(list(map(str,extracted)))),2)

    # Get random location to read data
    random_bit_location = np.random.RandomState(seed=secret_key*number_element).choice(frame_bytes_length,word_size*(number_element+1))
    random_bit_location = list(set(random_bit_location) - set(number_random_position))[:word_size*number_element]

    # Extract frame data
    finalExtraction = [] # Final extracted frame data
    #print(number_element,word_size)
    for i in range(0,(number_element*word_size)):
        finalExtraction.append(frame_bytes[random_bit_location[i]] & 1)

    # Extract bits and club them together to form word and then convert it to numbers
    frame_list = list(np.array(finalExtraction).reshape(number_element,word_size))
    number = list(map(lambda x: int("".join(list(map(str,x))),2),frame_list))
    random_frame_array=number
    #print(random_frame_array)

    #EXTRACT TEXT FROM VIDEO
    start=time.time()
    counter=0
    decoded_text=""
    for index in range(len(random_frame_array)):
        #print(random_frame_array[index]-1)

        subprocess.call(['ffmpeg','-i',videoFile,'-vf','select=eq(n\,{})'.format(random_frame_array[index]-1),
                         '-vframes','1','output{}.png'.format(counter)])
        prev_frame=cv2.imread("output{}.png".format(counter))

        counter+=1

        #print(prev_frame[0][0],prev_frame[0][1],prev_frame[0][2])
        #print(t_dict[random_frame_array[index]-1][0][0],t_dict[random_frame_array[index]-1][0][1],
        #      t_dict[random_frame_array[index]-1][0][2])

        corner=""
        c=-1
        for i in range(byte_len):
            if i%3==0:
                c+=1
            if(prev_frame[0,c,i%3]%2==0):
                corner+='0'
            else:
                corner+='1'
        c=2

        word_len=""
        for i in range(byte_len):
            if i%3==0:
                c+=1
            if(prev_frame[0,c,i%3]%2==0):
                word_len+='0'
            else:
                word_len+='1' 

        corner=int(corner,2)
        word_len=int(word_len,2)

        #print(corner,word_len)
        columns_needed=math.ceil(word_len*(byte_len+1)/3)

        if corner==1:
            row=0
            col=0
        if corner==2:
            row=0
            col=prev_frame.shape[1]-(columns_needed+1)
        if corner==3:
            row=prev_frame.shape[0]-1
            col=0
        if corner==4:
            row=prev_frame.shape[0]-1
            col=prev_frame.shape[1]-(columns_needed+1)

        #print(row,col)

        r=row
        if col!=0:
            c=col+1
        else:
            c=0

        subprocess.call(['ffmpeg','-i',videoFile,'-vf','select=eq(n\,{})'.format(random_frame_array[index]),
                         '-vframes','1','output{}.png'.format(counter)])
        curr_frame=cv2.imread("output{}.png".format(counter))
        counter+=1

        #print((byte_len+1)*3)
        for j in range(word_len):
            binary_string=""
            for i in range(byte_len):
                if i%3==0 and i!=0:
                    c+=1

                #print(i,c)
                if i%8==0 and i!=0:
                    #print(binary_string)
                    c+=1

                #print(i,r,c,frame_copy[r,c,i%3]%2)
                if(curr_frame[r,c,i%3]%2==0):
                    binary_string+='0'
                else:
                    binary_string+='1'
            c+=1
            decoded_text+=chr(int(binary_string,2))
        decoded_text+=" "
    decoded_text=decoded_text.strip().replace("- ","")
    print("\nDecoded text:{}".format(decoded_text))

    for f in glob.glob("output*.png"):
        os.remove(f)

    end=time.time()
    print(end-start)

    with open("static/decodedText.csv",'w') as file:
        file.write(decoded_text)

    return decoded_text
# In[8]:



# In[24]:


'''
counter = 0
for index in range(842):
    subprocess.call(
        ['ffmpeg', '-y', '-i', 'final_output.mkv', '-vf', 'select=eq(n\,{})'.format(index),
         '-vframes', '1', 'final_output{}.png'.format(counter)])
    subprocess.call(
        ['ffmpeg', '-y', '-i', 'output.mkv', '-vf', 'select=eq(n\,{})'.format(index),
         '-vframes', '1', 'output{}.png'.format(counter)])
    counter += 1
for index in range(842):
    before_frame = cv2.imread("final_output{}.png".format(index))
    after_f
'''

