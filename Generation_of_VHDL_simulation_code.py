# GENERATING VHDL CODE FOR SIMULATION OF FACE RECOGNITION IMPLEMENTATION

import sys
import numpy as np
from PIL import Image
import os

counter = 0
inputDir = os.path.join(os.getcwd(),'vstupne_data')
if not os.path.exists(inputDir):
    raise ValueError('Adresár vstupne_data neexistuje, vytvorte ho a vložte tam obrázok začínajúci sa na jednu z týchto hodnôt (gates,trump,zuckerberg) a byť vo formáte JPG - ' + inputDir)
for filename in os.listdir(inputDir):
    counter += 1;
if counter != 1:
    raise ValueError('V adresári môže byť len jeden súbor : ' + inputDir + 'Súbor sa musí začínať na jednu z týchto hodnôt (gates,trump,zuckerberg) a byť vo formáte JPG')

label = ""
def get_bin(x, n=0):
    return format(x, 'b').zfill(n)

for filename in os.listdir(inputDir):
    if filename.startswith('gates'):
        label = 'Gates - Output 0'
    if filename.startswith('trump'):
        label = 'Trump - Output 1'
    if filename.startswith('zuckerberg'):
        label = 'Zuckerberg - Output 2'
    tesbenchText = '--'+label+'\nlibrary IEEE;\nlibrary customlib;\nuse customlib.fixed_pkg.ALL;\nUSE IEEE.STD_LOGIC_1164.ALL;\n\nENTITY CONVOLUTIONAL_NETWORK_TB IS\nEND CONVOLUTIONAL_NETWORK_TB;\n\nARCHITECTURE behavior OF CONVOLUTIONAL_NETWORK_TB IS \nCOMPONENT TOP_MODULE\n\nPORT(\n    CLK_SOURCE_P : IN STD_LOGIC;\n    CLK_SOURCE_N : IN std_logic;\n    RST		: IN std_logic;\n    PIC_IN : std_logic;\n    OUTPUT_0 : out STD_LOGIC;\n    OUTPUT_1 : out STD_LOGIC;\n    OUTPUT_2 : out STD_LOGIC\n    );\nEND COMPONENT;\n\nCONSTANT F_POS : integer := 7;\nCONSTANT F_NEG : integer := 0;\n\nTYPE INPUT_DATA_TYPE is array(0 to 27,0 to 27) of std_logic_vector(F_POS downto F_NEG);\nCONSTANT INPUT_DATA : INPUT_DATA_TYPE :=\n(\n'
    fileFullPath = os.path.join(inputDir, filename)
    image = Image.open(fileFullPath).resize((28, 28)).convert('L')
    imageArray = np.array(image)
    for y in range(28):
        tesbenchText +='('
        for x in range(28):
            if x == 27:
                tesbenchText += '"' + str(get_bin(imageArray[y][x],8)) + '"'
            else:
                tesbenchText += '"'+str(get_bin(imageArray[y][x],8))+'",'
        if y == 27:
            tesbenchText += ')\n'
        else:
            tesbenchText += '),\n'
    tesbenchText += ');\n\n'
tesbenchText += 'SIGNAL CLK_P 		: std_logic := \'0\';\nSIGNAL CLK_N : std_logic := \'0\';\nSIGNAL RST 		: std_logic := \'0\';\nSIGNAL PIC_IN : std_logic :=\'0\';\nCONSTANT CLOCK_PERIOD	: time := 5 ns;\nSIGNAL ROW : integer := 0;\nSIGNAL COL : integer := 0;\nSIGNAL OUTPUT_0 : std_logic;\nSIGNAL OUTPUT_1 : std_logic;\nSIGNAL OUTPUT_2 : std_logic;\nconstant c_CLKS_PER_BIT : integer := 868;\nconstant c_BIT_PERIOD : time := 80 ns;\n\nBEGIN\n\nCONVOLUTIONAL_NETWORK_TOP_MODULE: TOP_MODULE \nPORT MAP(\n    CLK_SOURCE_P => CLK_P,\n    CLK_SOURCE_N => CLK_N,\n    RST => RST,\n    PIC_IN => PIC_IN,\n\n    OUTPUT_0 => OUTPUT_0,\n    OUTPUT_1 => OUTPUT_1,\n    OUTPUT_2 => OUTPUT_2\n);\n\n    clock_generation :process\n    begin\n        CLK_P <= \'1\';\n        CLK_N <= \'0\';\n        wait for CLOCK_PERIOD/2;\n        CLK_P <= \'0\';\n        CLK_N <= \'1\';\n        wait for CLOCK_PERIOD/2;\n    end process clock_generation;\n\n    uart: process\n    begin\n        wait for 3000ns;\n        PIC_IN <= \'1\';\n        wait for 3000ns;\n        for item in 0 to 783 loop\n            wait until rising_edge(CLK_P);\n            PIC_IN <= \'0\';\n            wait for c_BIT_PERIOD;\n\n            for ii in 0 to 7 loop\n            PIC_IN <= INPUT_DATA(ROW,COL)(ii);\n            wait for c_BIT_PERIOD;\n            end loop;\n\n            PIC_IN <= \'1\';\n            wait for c_BIT_PERIOD;\n            wait until rising_edge(CLK_P);\n                COL <= COL + 1;\n            if COL = 27 then\n                ROW <= ROW + 1;\n                COL <= 0;\n            end if;\n        end loop;\n        wait;\n    end process uart;\n\n    reset: process\n        begin\n        RST <=\'1\';\n        wait for 20 ns;\n        RST <=\'0\';\n        wait;\n    END PROCESS reset;\n    classify: process\n    begin\n        wait for 200 ns;\n        wait on OUTPUT_0,OUTPUT_1,OUTPUT_2;\n        IF OUTPUT_0 = \'1\' THEN\n            report "Klasifikovana osoba - gates";\n        END IF;\n        IF OUTPUT_1 = \'1\' THEN\n            report "Klasifikovana osoba - trump";\n        END IF;\n        IF OUTPUT_2 = \'1\' THEN\n            report "Klasifikovana osoba - zuckerberg";\n        END IF;\n        wait;\n    END PROCESS classify;\nEND;'

original_stdout = sys.stdout
with open('CONVOLUTIONAL_NETWORK_TB.vhd', 'w') as fileOutput:
    sys.stdout = fileOutput
    print(tesbenchText)
    sys.stdout = original_stdout
    print('Úspešne vygenerovaný súbor CONVOLUTIONAL_NETWORK_TB.vhd v adresári - ' + os.getcwd())