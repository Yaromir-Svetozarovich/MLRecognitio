from PIL import Image
img = Image.new('RGB', (32, 32), color=(37, 99, 235))
img.save('ui-prototype/img/favicon.ico', format='ICO')
print('favicon.ico created')