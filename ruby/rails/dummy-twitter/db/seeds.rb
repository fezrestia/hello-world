# This file should contain all the record creation needed to seed the database with its default values.
# The data can then be loaded with the rake db:seed (or created alongside the db with db:setup).
#
# Examples:
#
#   cities = City.create([{ name: 'Chicago' }, { name: 'Copenhagen' }])
#   Mayor.create(name: 'Emanuel', city: cities.first)

@user = User.new
@user.name = 'Ryo Suzuki'
@user.username = 'ryooopan'
@user.location = 'Kanagawa, Japan'
@user.about = 'hello'
@user.save

@user = User.new
@user.name = 'Shohei Aoki'
@user.username = 'moyahima'
@user.location = 'Tottori, Japan'
@user.about = 'world'
@user.save

