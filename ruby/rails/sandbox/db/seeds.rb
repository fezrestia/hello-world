# This file should contain all the record creation needed to seed the database with its default values.
# The data can then be loaded with the rails db:seed command (or created alongside the database with db:setup).
#
# Examples:
#
#   movies = Movie.create([{ name: 'Star Wars' }, { name: 'Lord of the Rings' }])
#   Character.create(name: 'Luke', movie: movies.first)

# Main Tables.
Platform.create(id: 1, name: "eSheep")
Platform.create(id: 2, name: "Rhine")
Platform.create(id: 3, name: "HB109")
Platform.create(id: 4, name: "RioGrande")
Model.create(id: 1, name: "Rachael")
Model.create(id: 2, name: "Dolly")
Model.create(id: 3, name: "Kurara")
Model.create(id: 4, name: "Kanna")
Model.create(id: 5, name: "Honami")
Model.create(id: 6, name: "Togari")
Model.create(id: 7, name: "Zeus")
Color.create(id: 1, name: "White")
Color.create(id: 2, name: "Black")
Color.create(id: 3, name: "Purple")
Color.create(id: 4, name: "Blue")
Color.create(id: 5, name: "Red")

# Relation Tables.
PlatformModelRel.create(id: 1, platform_id: 1, model_id: 1)
PlatformModelRel.create(id: 2, platform_id: 1, model_id: 2)
PlatformModelRel.create(id: 3, platform_id: 3, model_id: 3)
PlatformModelRel.create(id: 4, platform_id: 3, model_id: 4)
PlatformModelRel.create(id: 5, platform_id: 2, model_id: 5)
PlatformModelRel.create(id: 6, platform_id: 2, model_id: 6)
ModelColorRel.create(id:1, model_id: 1, color_id: 1)
ModelColorRel.create(id:2, model_id: 1, color_id: 2)
ModelColorRel.create(id:3, model_id: 2, color_id: 2)
ModelColorRel.create(id:4, model_id: 3, color_id: 3)
ModelColorRel.create(id:5, model_id: 3, color_id: 4)
ModelColorRel.create(id:6, model_id: 4, color_id: 1)
ModelColorRel.create(id:7, model_id: 5, color_id: 1)
ModelColorRel.create(id:8, model_id: 5, color_id: 2)
ModelColorRel.create(id:9, model_id: 5, color_id: 3)
ModelColorRel.create(id:10, model_id: 6, color_id: 1)
ModelColorRel.create(id:11, model_id: 6, color_id: 2)
ModelColorRel.create(id:12, model_id: 6, color_id: 3)

