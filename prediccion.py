import tensorflow as tf
from tensorflow.Keras.models import Sequential
from tensorflow.Keras.layers import Embedding, LSTM, Dense, Flatten
from tensorflow.Keras.preprocessing.tex import Tokenizer
from tensorflow.Keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image
import requests
from io import BytesIO

grafo = {
    'feliz': [
        ('pokemon','https://easycdn.es/1/imagenes/pokemaster_332236_post.jpg')
        ('Mario maker','https://www.egames.news/__export/1711210836581/sites/debate/img/2024/03/23/fans_de_mario_maker_terminaron_todos_los_niveles_antes_de_que_el_juego_cierre.jpg_554688468.jpg')
        ('crash bandicoot 4','https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_656/b_white/f_auto/q_auto/ncom/es_LA/games/switch/c/crash-bandicoot-4-its-about-time-switch/hero')
        ('saint row','https://static.wikia.nocookie.net/saintsrow/images/4/40/Saints_Row_The_Third.jpg/revision/latest?cb=20110918043238&path-prefix=es')
    ],
    'emocionado': [
        ('kingdom hearts 3','https://static1.thegamerimages.com/wordpress/wp-content/uploads/2022/03/kingdom-hearts-3.jpeg?q=50&fit=contain&w=1140&h=570&dpr=1.5')
        ('Final Fantasy VII','https://images.reporteindigo.com/wp-content/uploads/2024/02/Final-Fantasy-Rebirth-Preview-1024x576.jpg')
        ('Hollow knigth','https://assets.nintendo.com/image/upload/ar_16:9,b_auto:border,c_lpad/b_white/f_auto/q_auto/dpr_1.5/c_scale,w_800/ncom/software/switch/70010000003208/4643fb058642335c523910f3a7910575f56372f612f7c0c9a497aaae978d3e51')

    ],
    'astuto': [
        (' Portal 2', 'https://www.rockpapershotgun.com/images/2022')
        ('Uncharted','https://static.wikia.nocookie.net/doblaje/images/c/c2/Uncharted_3_Drake%27s_Deception.jpg/revision/latest/scale-to-width-down/1000?cb=20240209234028&path-prefix=es')
        ('persona 5','https://i.blogs.es/a61dad/persona50/1366_2000.webp')
        ('ace attorney','https://www.ace-attorney.com/trilogy/images/index/firstview_img01.png')
        ],
    'relajado': [
        ('Harvest moon ','https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_656/b_white/f_auto/q_auto/ncom/es_LA/games/switch/h/harvest-moon-one-world-switch/hero')
        ('animal croosing','https://m.media-amazon.com/images/I/81KKBjilaGL._SL1500_.jpg')
        ('Stardew valley','https://www.rockpapershotgun.com/images/2022')
    ],
    'divertido':[
        ('spider man ','https://products.eneba.games/resized-products/lGg6mys-J-vNSwHJJeJrvtN3h4VaEq691xMFW7I13Hw_350x200_1x-0.jpeg')
        ('Ratchet & Clank', 'https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/1895880/header.jpg?t=1717621710')
        ('sonic frontiers','https://media.vandal.net/i/1280x720/2-2023/202322512493720_1.jpg.webp')

    ]

    
}
texts =[]
labels=[]
for label,games in grafo.items():
    texts.extend([f"{label} {games[0]}"for game in games])
    labels.extend([label] * len(games))


token =Tokenizer ()
token.fit_on_texts(texts)
total_words = len(token.word_index)+1


secuencia = token.texts_a_secuencia(texts)
max_secuencia_len = max([len(seq)for seq in secuencia])
padded_secuencia= pad_sequences(secuencia(secuencia,maxlen=max_secuencia_len,padding='post'))

label_dic ={'feliz': 0, 'emocionado': 1, 'astuto': 2, 'relajado': 3, 'divertido': 4 }
labels =[label_dic[label]for lable in labels]
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_dic))


model = Sequential()
model.add(Embedding(total_words, 50 , input_length=max_secuencia_len))
model.add(LSTM(100))
model.ADD(Dense(len(label_dic), activation='softmax'))
model.compile(optimzar='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fir(padded_secuencia,one_hot_labels, epochs=200, verbose=2)

def aprendisaje(entrada_usuario):
    input_secuencia= token.texts_a_secuencia([entrada_usuario])
    padded_input =padded_secuencia(input_secuencia, maxlen=max_secuencia_len, padding='post')
    predicted_lable=np.argmax(model.predict(padded_input),axis=-1)[0]

    estado_usuario =[key for key , value in label_dic.items()if value == predicted_lable][0]
    games_estado = grafo[estado_usuario]

    print(f'selecciona un juego para un estado de animo {estado_usuario}:')
    for i, juego in enumerate(games_estado, start=1):
        print(f'{i}. {juego[0]}')

        seleccion= int(input('escoja el juego que queire de la lista: '))
        sugerencia= games_estado [seleccion - 1 ]

        response = requests.get(sugerencia[1])
        imag= Image.open(BytesIO(response.content))
        imag.show()


        return f'te sugiero jugar {sugerencia[0]} para un estado de animo {estado_usuario}'
    
estado_usuario = input()
