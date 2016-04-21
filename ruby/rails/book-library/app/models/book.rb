class Book < ActiveRecord::Base
    belongs_to :user
    has_many :taggings
    has_many :tags, through: :taggings

    scope :tagged_recommended, -> { joins(:tags).where(tags: {name: 'recommended'}) }

    after_save :increment_user_books_size

private
    def increment_user_books_size
#        self.user.increment!(:books_size)
    end

end

