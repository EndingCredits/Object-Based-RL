# Example VGDL description text
# The game dynamics are specified as a paragraph of text

aliens_game = """
BasicGame block_size=10
    SpriteSet
        background > Immovable img=oryx/space1 hidden=True
        base    > Immovable    color=WHITE img=oryx/planet
        avatar  > FlakAvatar   stype=sam img=oryx/spaceship1
        missile > Missile
            sam  > orientation=UP    color=BLUE singleton=True img=oryx/bullet2
            bomb > orientation=DOWN  color=RED  speed=0.5 img=oryx/bullet2
        alien   > Bomber       stype=bomb   prob=0.05  cooldown=3 speed=0.8
            alienGreen > img=oryx/alien3
            alienBlue > img=oryx/alien1
        portal  > invisible=True hidden=True
        	portalSlow  > SpawnPoint   stype=alienBlue  cooldown=16   total=20
        	portalFast  > SpawnPoint   stype=alienGreen  cooldown=12   total=20

    LevelMapping
        . > background
        0 > background base
        1 > background portalSlow
        2 > background portalFast
        A > background avatar

    TerminationSet
        SpriteCounter      stype=avatar               limit=0 win=False
        MultiSpriteCounter stype1=portal stype2=alien limit=0 win=True


    InteractionSet
        avatar  EOS  > stepBack
        alien   EOS  > turnAround
        missile EOS  > killSprite

        base bomb > killSprite
        base sam > killSprite scoreChange=1

        base   alien > killSprite
        avatar alien > killSprite scoreChange=-1
        avatar bomb  > killSprite scoreChange=-1
        alien  sam   > killSprite scoreChange=2
"""

# the (initial) level as a block of characters
aliens_level = """
1.............................
000...........................
000...........................
..............................
..............................
..............................
..............................
....000......000000.....000...
...00000....00000000...00000..
...0...0....00....00...00000..
................A.............
"""

