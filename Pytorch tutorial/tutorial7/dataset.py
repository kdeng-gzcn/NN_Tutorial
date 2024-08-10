# data pre-processing is super important

class DogsVSCats():
    IMG_SIZE = 50 # want to normailize all the images, make it 50 * 50 pixels

    # note that data augment is really common in CV

    CATS = '../../../dataset/kagglecatsanddogs_5340/PetImages/Cat'
    DOGS = '../../../dataset/kagglecatsanddogs_5340/PetImages/Dog'

    LABELS = {CATS: 0, DOGS: 1}

    training_data = []

    # keep balanced data is super important 
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS: # iterating the keys
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # color doesnt matter in clasifying the cat and dog
                    # note that we want to make our data and nn as small as possible
                    img = cv2.resize(img, [self.IMG_SIZE, self.IMG_SIZE]) # reshape in 50 * 50
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    # count our data
                    if label == self.CATS:
                        self.catcount += 1
                    if label == self.DOGS:
                        self.dogcount += 1

                except Exception as e:
                    # print(str(e))
                    pass


        np.random.shuffle(self.training_data) # shuffle is inplace
        np.save('training_data.npy', self.training_data)

        print('cats:', self.catcount)
        print('dogs:', self.dogcount)