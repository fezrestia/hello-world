class CreatePlatforms < ActiveRecord::Migration[5.1]
  def change
    create_table :platforms do |t|

      t.string :name

      t.timestamps
    end
  end
end
