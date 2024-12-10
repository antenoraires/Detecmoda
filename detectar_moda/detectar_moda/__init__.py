from detectar_moda import control

class Pipeline:
    def __init__(self, img_height, img_width, batch_size):
        self.train_dir = 'assents/Train'
        self.validation_dir = 'assents/Validation'
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.good_exts  =["bmp", "gif", "jpeg", "png","jpg"]
    
    def train(self):  # Método 'train' em minúsculas, conforme convenção
        result = control.train_model(self.train_dir,
                             self.img_height,
                             self.img_width, self.batch_size)
        return result
    
    def validation(self):  # Método 'validation' corrigido
        result = control.validation_model(self.validation_dir, 
                                  self.img_height, 
                                  self.img_width, self.batch_size)
        return result
    
    def check_img(self):
        bad_images, bad_ext = control.check_images(self.train_dir, self.good_exts)
        return  bad_images, bad_ext
