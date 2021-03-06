(function(){

  var thread_id = window.location.hash != '' ? window.location.hash.substr(1, 2) : 0; 

  Dom.get('j-btn-start').onclick = function(){
    var space = {
      playerX_space: [-0.04, 0.04],
      speed_space: [3.0, 10.0],
    };
    // tell server the range of action, for normalization
    // for avoiding exceptions in replay buffer on server
    // START_FRAME = true;
    Game.run(gameParams);
    // speed = 50*100;
  };

  Dom.get('j-btn-stop').onclick = function(){
    Game.stop();
  };

  Dom.get('j-btn-train').onclick = function(){
    isTraining = !isTraining;
    keyFaster = false;
    this.innerText = 'train:(' + isTraining + ')';
  }

  /*----------- the above is controller----------------------------------*/
  var isTraining = false;
  var timeIntervalID = null;
  // dynamically create a smaller canvas for preview
  var smallCanvas = document.createElement('canvas');
  var smallCtx = smallCanvas.getContext('2d');
  var smallImage = new Image();
  var smallWidth = 84;
  var smallHeight = 84;
  smallCanvas.width = smallWidth;
  smallCanvas.height = smallHeight;
  smallCanvas.zIndex = '100';
  Dom.get('preview').appendChild(smallCanvas);

  function capture() {
    // scale the snapshot of main canvas to a smaller one
    smallCtx.drawImage(canvas, 0, 0, smallWidth, smallHeight);
    var prefix = 'data:image/png;base64,';
    return smallCanvas.toDataURL().substr(prefix.length);
  }

  function getReward() {
    var COLLISION_COST = -1.0;
    var OFF_ROAD_COST = -1.0;
    var LANE_PENALTY = 0.5;

    if (COLLISION_OCCURED) {
      return COLLISION_COST;
    }
    var pos = Math.abs(playerX);
    if (pos > 1.0) {
      return OFF_ROAD_COST;
    }

    if(speed <= 10){
      return -0.1; // to slow
    }

    var inLane = pos <= 0.1 || (pos >= 0.6 && pos <= 0.8)
    var penalty = inLane ? 1 : LANE_PENALTY;
    return penalty * (speed / maxSpeed);
  }



//=========================================================================
// THE GAME LOOP
//=========================================================================
  
  var sampleCount = 0;
  var gameParams = {
    canvas: canvas, render: render, update: update, stats: stats, step: step,
    images: ["background", "sprites"],
    keys: [
      { keys: [KEY.LEFT,  KEY.A], mode: 'down', action: function() { keyLeft   = true;  } },
      { keys: [KEY.RIGHT, KEY.D], mode: 'down', action: function() { keyRight  = true;  } },
      { keys: [KEY.UP,    KEY.W], mode: 'down', action: function() { keyFaster = true;  } },
      { keys: [KEY.DOWN,  KEY.S], mode: 'down', action: function() { keySlower = true;  } },
      { keys: [KEY.LEFT,  KEY.A], mode: 'up',   action: function() { keyLeft   = false; } },
      { keys: [KEY.RIGHT, KEY.D], mode: 'up',   action: function() { keyRight  = false; } },
      { keys: [KEY.UP,    KEY.W], mode: 'up',   action: function() { keyFaster = false; } },
      { keys: [KEY.DOWN,  KEY.S], mode: 'up',   action: function() { keySlower = false; } }
    ],

    ready: function(images) {
      background = images[0];
      sprites    = images[1];
      reset();
      Dom.storage.fast_lap_time = Dom.storage.fast_lap_time || 180;
      // updateHud('fast_lap_time', formatTime(Util.toFloat(Dom.storage.fast_lap_time)));
    },

    afterUpdate: function(dt){ 
      // console.log(keyFaster);
      // if collision or off-road occurs, restart the game
      var terminal = false;
      if (COLLISION_OCCURED || Math.abs(playerX) > 1.0){
        terminal = true; 
      }

      sampleCount += 1;
      if(!terminal && sampleCount < 2){
        return;
      }
      sampleCount = 0;

      var img = capture();
      var reward = getReward();

      var data = {
        img: img,
        reward: reward,
        terminal: terminal,
        start_frame: START_FRAME,
        thread_id: thread_id
      }

      if(isTraining){
        // socket.emit('message', data);
        sendToServer(data);
        if (START_FRAME){
          START_FRAME = false;
        }
        Game.stop();
      }

      lastPlayerX = playerX;
      lastSpeed = speed;

      if(terminal){
        Game.restart();
      }

    }
  };

  function sendToServer(data){
    $.ajax({
      url: '/train',
      type: 'post',
      data: data,
      dataType: 'json',
      success: function(ret){
        keyLeft = ret['keyLeft'];
        keyRight = ret['keyRight'];
        keyFaster = ret['keyFaster'];
        keySlower = ret['keySlower'];
        if (!isTraining){
          keyFaster = false;
        }
      }
    });
  }



})();