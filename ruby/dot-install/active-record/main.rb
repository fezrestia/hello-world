require 'active_record'
require 'logger'

ActiveRecord::Base.establish_connection(
    "adapter" => "sqlite3",
    "database" => "./blog.db"
)

ActiveRecord::Base.logger = Logger.new(STDOUT)

class Post < ActiveRecord::Base
    scope :top3, -> {
        order("created_at").limit(3)
    }

    validates :title, :presence => true
    validates :body, :length => {:minimum => 5 }

    has_many :comments, :dependent => :destroy

end

class Comment < ActiveRecord::Base
    belongs_to :post
end


#post = Post.new(:title => "title1", :body => "hellp!")

#post = Post.new
#post.title = "title2"
#post.body = "hello2"

#post = Post.new do |p|
#    p.title = "title3"
#    p.body = "bello3"
#end

#post.save

#Post.create(:title => "title4", :body => "hello4")

#p Post.all

#p Post.first
#p Post.last.title

#p Post.find(3)

#p Post.find_by_title("title2")

#p Post.find_by_title_and_id("title2", 2)

#p Post.where(:title => "title1", :id => 1)
#p Post.where("title = ? and id = ?", "title1", 1)
#p Post.where("title = :title and id = :id", {:title => "title1", :id => 1})

#p Post.where("id > ?", 2)

#p Post.where("body like ?", "%ell%")

#p Post.where(:id => 1..3)
#p Post.where(:id => [1, 3])

#p Post.order("id desc").limit(3)

#p Post.top3


#p Post.where(:title => "title5").first_or_create
#Post.where(:title => "title7").first_or_create do |p|
#    p.body = "hello7"
#end


#post = Post.find(1)
#post.title = "new title 1"
#post.save

#post.update_attribute(:title, "new 2 title 1")

#post.update_attributes(:title => "nnn", :body => "hhh")

#Post.where(:id => 1..2).update_all(:title => "nnn2", :body => "hhh2")



#Post.where(:id => 1..2).delete_all

#Post.find(3).destroy


#post = Post.new(:body => "123")

#if !post.save
#    p post.errors.messages
#end




post = Post.find(1);
post.comments.each do |comment|
    p comment.body
end


p Post.all
p Comment.all

Post.find(1).destroy

p Post.all
p Comment.all






#p Post.all

