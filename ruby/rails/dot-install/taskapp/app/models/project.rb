class Project < ActiveRecord::Base

    has_many :tasks

    validates :title,
            presence: { message: "Input Title" },
            length: { minimum: 3, message: "Too Short <=3" }

end

