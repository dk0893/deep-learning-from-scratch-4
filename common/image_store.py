import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

class ImageStore:
    
    ope   = None
    dpath = None
    fpath = None
    
    artists_step  = []
    artists_state = []
    animation     = False
    
    step  = 0
    phase = 0
    cnt   = 0
    
    debug = False
    
    def init( ope=None, dpath=None, fpath=None, debug=False ):
        
        ImageStore.ope   = ope
        ImageStore.dpath = dpath
        ImageStore.fpath = fpath
        ImageStore.debug = debug
        
        if ImageStore.ope == "ani_step" or ImageStore.ope == "ani_state":
            ImageStore.animation = True
        elif ImageStore.ope == "im_step" or ImageStore.ope == "im_state":
            os.makedirs( ImageStore.dpath, exist_ok=True )
    
    def st_step( env, V, pi ):
        
        if ImageStore.ope == "im_step" or ImageStore.ope == "ani_step" or ImageStore.ope is None:
            frame = env.render_v(V, pi, title=f"step={ImageStore.step}")
            ImageStore.artists_step.append( frame )
        
        if ImageStore.ope == "im_step":
            fpath = os.path.join( ImageStore.dpath, f"policy_iter_step_{ImageStore.step}.png" )
            plt.savefig( fpath )
            plt.close()
            
            print( f"save image: {fpath}" )
        
        ImageStore.step += 1
        ImageStore.phase = 0
    
    def st_state( env, V, pi, state ):
        
        if ImageStore.ope == "im_state" or ImageStore.ope == "ani_state":
            frame = env.render_v( V, pi, title=f"step={ImageStore.step} phase={ImageStore.phase} state={state}" )
            ImageStore.artists_state.append( frame )
            
            if ImageStore.ope == "im_state":
                fpath = os.path.join( ImageStore.dpath, f"policy_iter_step{ImageStore.step}_phase{ImageStore.phase:02d}_state_{state}.png" )
                plt.savefig( fpath )
                plt.close()
                print( f"save image: {fpath}" )
            
            ImageStore.cnt += 1
            if ImageStore.cnt % np.prod(env.shape) == 0:
                ImageStore.phase += 1
            
            if ImageStore.debug:
                if ImageStore.ope == "ani_state" and ImageStore.phase == 1:
                    ImageStore.ope = "ani_end"
    
    def output( fig ):
        
        if ImageStore.ope == "ani_step" or ImageStore.ope == "ani_state" or ImageStore.ope == "ani_end":
            artists = ImageStore.artists_step if ImageStore.ope == "ani_step" else ImageStore.artists_state
            interval = 2000 if ImageStore.ope == "ani_step" else 500
            anim = ArtistAnimation( fig, artists, interval=interval )
            anim.save( ImageStore.fpath )
            
            print( f"save animation: {ImageStore.fpath}" )
        
        if ImageStore.ope == "im_state" or ImageStore.ope == "ani_state":
            print( f"ImageStore.cnt={ImageStore.cnt}" )

