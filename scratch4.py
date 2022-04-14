import base64

image = open('test.jpeg', 'rb')

image_read = image.read()

image_64_encode = base64.b64encode(image_read)
image_64_decode = base64.b64decode(image_64_encode)
image_result = open('deer_decode.gif', 'wb') # create a writable image and write the decoding result
image_result.write(image_64_decode)