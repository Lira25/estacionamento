import cv2 
import numpy as np

# Definição das coordenadas das vagas de estacionamento.
vaga1 = [1, 89, 108, 213]
vaga2 = [115, 87, 152, 211]
vaga3 = [289, 89, 138, 212]
vaga4 = [439, 87, 135, 212]
vaga5 = [591, 90, 132, 206]
vaga6 = [738, 93, 139, 204]
vaga7 = [881, 93, 138, 201]
vaga8 = [1027, 94, 147, 202]

# Lista que contém todas as vagas, permitindo iteração em sequência.
vagas = [vaga1,vaga2,vaga3,vaga4,vaga5,vaga6,vaga7,vaga8]

# Abertura do vídeo que será processado.
video = cv2.VideoCapture('video.mp4')

# Loop principal para processar cada quadro do vídeo.
while True:
    # Leitura de um quadro do vídeo.
    check,img = video.read()

    # Conversão da imagem para escala de cinza.
    imgCinza = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Aplicação de threshold adaptativo para binarizar a imagem (convertendo para preto e branco).
    imgTh = cv2.adaptiveThreshold(imgCinza,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)

    # Aplicação de filtro de mediana para suavizar a imagem e reduzir ruídos.
    imgBlur = cv2.medianBlur(imgTh,5)

    # Criação de um kernel (matriz) para dilatação da imagem.
    kernel = np.ones((3,3),np.int8)

    # Aplicação de dilatação para aumentar as regiões brancas (as áreas de interesse).
    imgDil = cv2.dilate(imgBlur,kernel)

    # Inicialização do contador de vagas abertas.
    qtVagasAbertas = 0

    # Iteração sobre todas as vagas definidas.
    for x,y,w,h in vagas:
        # Recorte da região da imagem dilatada correspondente a uma vaga específica.
        recorte = imgDil[y:y+h,x:x+w]

        # Contagem do número de pixels brancos na região recortada.
        qtPxBranco = cv2.countNonZero(recorte)

        # Exibição da quantidade de pixels brancos diretamente na imagem original.
        cv2.putText(img,str(qtPxBranco),(x,y+h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        # Verificação se a quantidade de pixels brancos é maior que um certo limiar (3000).
        if qtPxBranco > 3000:
            # Se for maior, desenha um retângulo vermelho ao redor da vaga (indica que está ocupada).
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        else:
            # Caso contrário, desenha um retângulo verde (indica que a vaga está livre).
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            qtVagasAbertas += 1  # Incrementa o contador de vagas abertas.

    # Desenha um retângulo azul na imagem para exibir o número de vagas livres.
    cv2.rectangle(img,(90,0),(415,60),(255,0,0),-1)

    # Exibe o número de vagas livres no retângulo azul.
    cv2.putText(img,f'LIVRE: {qtVagasAbertas}/8',(95,45),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),5)

    # Exibe a imagem original processada (com os retângulos coloridos e informações de vagas).
    cv2.imshow('video',img)

    # Exibe a imagem binarizada e dilatada (usada para detecção).
    cv2.imshow('video TH', imgDil)

    # Aguarda por 10ms antes de processar o próximo quadro.
    cv2.waitKey(10)
