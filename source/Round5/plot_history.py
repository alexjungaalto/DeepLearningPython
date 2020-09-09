def plot_history(history):
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    #-----------------------------------------------------------
    # Retrieve results on training and validation data sets
    # for each training epoch
    #-----------------------------------------------------------
    if isinstance(history, pd.DataFrame):
        
        acc      = history['accuracy']
        val_acc  = history['val_accuracy']
        loss     = history['loss']
        val_loss = history['val_loss']
        epochs   = range(1,len(acc)+1) 
    else:
        
        acc      = history.history['accuracy']
        val_acc  = history.history['val_accuracy']
        loss     = history.history['loss']
        val_loss = history.history['val_loss']
        epochs   = range(1,len(acc)+1) 
        
    
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
    
    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    ax1.plot(epochs, acc,  label='Training accuracy')
    ax1.plot(epochs, val_acc,  label='Validation accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    # ax1.set_ylim(0.5,1)
    
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------

    ax2.plot(epochs, loss,  label='Training Loss')
    ax2.plot(epochs, val_loss,  label='Validation Loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    # ax2.set_ylim(0,2)
    ax2.legend()

    fig.tight_layout()
    plt.show()