import pandas as pd
import csv
from intbitset import intbitset
import pickle



class Data():
    """
    """
    def __init__(self):
        pass
    
    
    
    def load_recsys_data(self, foldername):
        """
        """
        
        ######################        
        ######################
        # LOAD matrix
        with open(foldername + "matrix-factorized.p", 'rb') as handle:
            matrix_factorized = pickle.load(handle)
            
        self.matrix_factorized = matrix_factorized
        
        ######################        
        ######################
        # LOAD MAPPING uid-mid
        file_mapping_uid_mid = open(foldername + "mapping_uid_mid.tsv", "r")
        csv_loader = csv.reader(file_mapping_uid_mid, delimiter="\t")
        
        mapping_users_to_rated_items = {}
        for row in csv_loader:
            uid = int(row[0])
            lst_ = row[1:]
            rated_items = set([int(mid) for mid in lst_])
            mapping_users_to_rated_items[uid] = rated_items

        self.mapping_users_to_rated_items = mapping_users_to_rated_items
        file_mapping_uid_mid.close()
        
        
        ######################        
        ######################
        # LOAD mid genre
        file_mapping_mid_genre = open(foldername + "mapping_mid_genre.tsv", "r")
        csv_loader = csv.reader(file_mapping_mid_genre, delimiter="\t")

        
        self.groups = set()
        # mapping genre to mid
        # mapping mid to genre
        self.mapping_genre_to_movie = {}
        self.mapping_movie_to_genre = {}
        for row in csv_loader:
            
            mid = int(row[0])
            genre = row[1]
            self.mapping_movie_to_genre[mid] = genre
            
            if genre not in self.groups:
                
                self.groups.update([genre])
                self.mapping_genre_to_movie[genre] = set()
            
            self.mapping_genre_to_movie[genre].update([mid])
            
        file_mapping_mid_genre.close()
        
        
    
    
    def initialize_user_recsys(self, uid):
        """
        """
        
        self.mapping_id_to_attributes = {}
        for mid in self.mapping_movie_to_genre:
            #if mid not in self.mapping_users_to_rated_items[uid]:
            self.mapping_id_to_attributes[mid] = self.mapping_movie_to_genre[mid]
        
        #self.mapping_groups = {}
        #for genre in self.mapping_genre_to_movie:
        #    self.mapping_groups[genre] = self.mapping_genre_to_movie[genre] - self.mapping_users_to_rated_items[uid]
        self.mapping_groups = self.mapping_genre_to_movie
            
        
            
        ######################        
        ######################
        # proportions needed
        self.proportions = {}
        tot = len(self.mapping_id_to_attributes)
        for i in self.mapping_groups:
            self.proportions[i] = round(1.*len(self.mapping_groups[i])/tot, 4)
            
            
        # items x items       
        self.VxV = self.matrix_factorized["HxH"]
        # users x items
        self.VxU = self.matrix_factorized["WxH"][uid,:]


        """
        print(len(self.mapping_movie_to_genre),
              len(self.mapping_id_to_attributes), 
              len(self.mapping_users_to_rated_items[uid]))
        """ 
            
    def initialize_profiles(self, filename, attribute):
        """
        """
        if "pokec" in filename:
            inputprofile = open(filename, "r")
            inputreader = csv.reader(inputprofile, delimiter="\t")
            self.mapping_id_to_attributes = {}
            self.groups = set()
            header = next(inputreader)
            for row in inputreader:
                id_ = int(row[0])
                attr_ = row[1]
                self.mapping_id_to_attributes[id_] = attr_
                if attr_ not in self.groups:
                    self.groups.update([attr_])
            
        else:
            profiles = pd.read_csv(filename, sep="\t",index_col=0)
            self.mapping_id_to_attributes = profiles.to_dict()[attribute]

        
            self.groups = set(profiles[attribute].unique())

        mapping_attributes_to_id = {}
        for el in self.mapping_id_to_attributes:
            attr = self.mapping_id_to_attributes[el]
            if attr not in mapping_attributes_to_id:
                mapping_attributes_to_id[attr] = set()
            mapping_attributes_to_id[attr].update([el])

        self.mapping_groups = mapping_attributes_to_id
        
        # proportions needed
        self.proportions = {}
        tot = len(self.mapping_id_to_attributes)
        for i in self.mapping_groups:
            self.proportions[i] = round(1.*len(self.mapping_groups[i])/tot, 4)
            
            
        
        
    def mapping_neighbors(self, filename_edgelist):
        """
        """
        
        inputfile = open(filename_edgelist, "r")
        tsvreader = csv.reader(inputfile, delimiter="\t")
        
        header=next(tsvreader)
        
        
        count = 0
        self.neighbors = {}
        if self.directed == True:
            for row in tsvreader:
                source = int(row[0])
                target = int(row[1])
                if source not in self.neighbors:
                    
                    #self.neighbors[source] = intbitset(rhs=500000)
                    self.neighbors[source] = set()
                    
                self.neighbors[source].update([target])
                
                # threshold
                count +=1
                if count == self.threshold:
                    break

        if self.directed == False:
            for row in tsvreader:
                source = int(row[0])
                target = int(row[1])
                
                # add source <-> neighs
                if source not in self.neighbors:
                    
                    #self.neighbors[source] = intbitset(rhs=500000)
                    self.neighbors[source] = set()
                
                self.neighbors[source].update([target])

                # add target <-> neighs
                if target not in self.neighbors:
                    
                    #self.neighbors[target] = intbitset(rhs=500000)
                    self.neighbors[target] = set()
                    
                self.neighbors[target].update([source])
                
                # threshold
                count +=1
                if count == self.threshold:
                    break
                    
        self.isolated = 0         
        for n in self.mapping_id_to_attributes:
            if n not in self.neighbors:
                #self.neighbors[n] = intbitset(rhs=500000)
                self.neighbors[n] = set()
                self.isolated +=1
        
                
                
                
                
