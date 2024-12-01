import imageio
face = imageio.imread(file)
face = cv2.resize(face, (128,128) )
faces = [face]
X=np.squeeze(faces)
X = X.astype('float32')
X /= 255
