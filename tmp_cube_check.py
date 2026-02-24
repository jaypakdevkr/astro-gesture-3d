from ursina import Ursina, Entity, camera
app = Ursina(borderless=False)
camera.position=(0,0,-6)
cube = Entity(model='cube', scale=1.5)
print('cube model type', type(cube.model))
print('cube model', cube.model)
if hasattr(cube.model, 'vertices'):
    print('verts', len(cube.model.vertices))
print('children of scene', scene.children)
for _ in range(2):
    app.step()
app.destroy()
