# This is installation guide for Data Analysis Using Flask on Mac labtops.

## Author: Fu-Chun Hsu

## Installation of Data Analysis Using Flask

* Hardware

  * Mac Labtop

* Software

  

  1. Download the Conda:

       * [Miniconda installer for macOS](https://conda.io/miniconda.html)

       * [Anaconda installer for macOS](https://www.anaconda.com/download/)

       * Install:
            
           * Miniconda---In your Terminal window, run:

             ``` Miniconda3-latest-MacOSX-x86_64.sh ```

       * Anaconda---Double-click the ``.pkg`` file.

  2. Setup the environment
 
     1. Use the environment.yml sent and
    
            $ conda env create --name demo -f environment.yml
     2. Activate the environment
    
            $ Source activate demo
  3. Install Dlib 
      
      Install dlib for as share library.
          
         $ brew install dlib
         $ pip install dlib

        
            
  4. Install required python packages
  
         $ pip install flask numpy requests
         
  5. Run app

         $ python main.py

  6. Go to local website : [link](http://0.0.0.0:5000/)