# Mask-Rcnn-Tensorflow-2.0

Now a days tensorflow 2.0 is taking over the world and object detection algorithms are the things in which researchers, students and indutries are intrested alot. So
I decides to make a mask-rcnn compatible with tensorflow 2.0. I took help from alot blogs tensorflow 2.0 documentations and stack-overflow. Now a days its really 
complicated to made a tensorflow code which is compatible with nvidia 3000 series. So I also made some changes to make this code run on my 3070rtx. So tighten your 
seat belts, we are going to start our drive.

# Introduction:
Main aim of this mrcnn git is to train MRCNN to detect
* Eye-Bags
* Wrinkles
* Pimples
* Glasses

# Making Dataset.
Downloaded Images of respective problem from google and annotate them with the help of [VIA Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html)

* For Annotations select the following Settings
    * Open the VIA annotator"
    * Remove files from images
    * Delete all of the attributes 
        * Go to the Region Attributes (At the bottom)
        * Write name of Attrubute in (attribute name field)
        * click on '-' to delete
        * Perform the steps above again and delete all of the attributes
        <img src="image_augmentation/extras/annotations_SS/remove_attributes.png"  width="200" height="200">
    * Add Attributes
        * First select polygons
            * <img src="image_augmentation/extras/annotations_SS/select_polygon.png"  width="300" height="100">
        * Now add 1 attribute name it as class 
        * Select type checkbox
        * Add first id as empty string ''
        * Add the other objects you want to annotation
            * Like I add Wrinkles
            <img src="/image_augmentation/extras/annotations_SS/add_attributes.png"  width="150" height="250">
        * Add files
            <img src="/image_augmentation/extras/annotations_SS/add_images.png"  width="150" height="250">
        * Start annotation

* Pimples
        
     <img src="/mrcnn_training/extras/pimples.png"  width="200" height="200">
* Wrinkles 
    
    <img src="/mrcnn_training/extras/wrinkles.png"  width="200" height="200">
* EyeBags 
    
    <img src="/mrcnn_training/extras/eyebags.JPG"  width="200" height="200">


## Augmentation:

After doing all the annotation things I rotate the image and polygons at every angle (0, 360) you can easily use my notebook to get done with your annotations. Its really complicated to use sin and cos to rotate a point with respect to the image. lol

## Training

After doing the augmentations just give path of the directory where you augmentated the data. If you have done augmentation from the notebook above than your augmentation will probably be on the path where json is located.

You just need to give the augmented images here.



<br>

## Evaluation

After training you have to open predict notebook. Specify weights path and testing images path and tada your predictions will be their in just few seconds.

**You need to set Generic Config according to train Generic Config and dont forget to give number of classes**

<br>
<br>



### Prediction





## Feel free to ask any question Thank you

<br>
<br>

# Author 

* Sohaib Anwaar  : [Portfolio](https:www.sohaibanwaar.com)
* gmail          : sohaibanwaar36@gmail.com
* linkedin       : [Have Some Professional Talk here](https://www.linkedin.com/in/sohaib-anwaar-4b7ba1187/)
* Stack Overflow : [Get my help Here](https://stackoverflow.com/users/7959545/sohaib-anwaar)
* Kaggle         : [View my master-pieces here](https://www.kaggle.com/sohaibanwaar1203)